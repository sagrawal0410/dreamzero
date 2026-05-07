"""Stage 5 — Allocator model + dataset for amortized importance prediction.

Self-contained (no DreamZero imports): the model takes a single observation
(image + proprioception + instruction + action history) and predicts a 2-D
importance map at the same spatial shape as the Stage-1 action-causal heatmap.

The training target is the Stage-1 perturbation-derived heatmap (averaged over
the T latent axis to a 2-D map). Training drives the allocator to mimic that
expensive perturbation signal from cheap, real-time inputs.

This module exposes:

    Allocator              — small CNN+MLP, configurable input ablations.
    PerturbMapDataset      — yields (image, proprio, lang_emb, history,
                              target_heatmap, meta) per example.
    HashLanguageEmbedder   — tiny dep-free instruction embedder.
    save_predictions_npz   — dump predicted maps in Stage-1's `heatmaps.npz`
                              layout so downstream stages (3/4) can re-use them.

Importable from `train_allocator.py` and `analyze_allocator.py`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger("stage5.allocator_model")


# ============================================================================
# Lightweight language embedder (no extra deps)
# ============================================================================


class HashLanguageEmbedder:
    """Deterministic dependency-free instruction embedder.

    Each token of the instruction is hashed to an integer seed; the seed
    drives a Gaussian projection of dimension `dim`. The instruction
    embedding is the mean of its token embeddings (a hashing-trick BoW).

    Pros: no external deps, deterministic, fast.
    Cons: doesn't capture semantics — replace with CLIP text or
    sentence-transformers in production. Used here to keep Stage-5 training
    a pure CPU/GPU job with no extra package install.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)
        self._cache: dict[str, np.ndarray] = {}

    def _seed_from_token(self, tok: str) -> int:
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def embed(self, instruction: str) -> np.ndarray:
        if instruction in self._cache:
            return self._cache[instruction]
        text = (instruction or "").lower().strip()
        if not text:
            v = np.zeros(self.dim, dtype=np.float32)
            self._cache[instruction] = v
            return v
        toks = [t for t in text.replace("/", " ").replace(",", " ").split() if t]
        if not toks:
            v = np.zeros(self.dim, dtype=np.float32)
            self._cache[instruction] = v
            return v
        vecs = []
        for tok in toks:
            rng = np.random.default_rng(self._seed_from_token(tok))
            vecs.append(rng.standard_normal(self.dim).astype(np.float32))
        v = np.mean(np.stack(vecs, axis=0), axis=0)
        self._cache[instruction] = v
        return v


# ============================================================================
# Dataset
# ============================================================================


@dataclass
class ExampleSpec:
    example_id: str
    task_group: str
    role: str
    episode_index: int
    frame_index: int
    image_path: Path
    proprio: np.ndarray            # flat float32 vector
    instruction: str
    prev_action_chunk: np.ndarray  # (history_len, action_dim) zero-padded
    target_heatmap: np.ndarray     # (target_h, target_w)


def _flatten_state(entry: dict[str, Any]) -> np.ndarray:
    s = np.asarray(entry.get("state_at_t", []), dtype=np.float32).reshape(-1)
    if s.size == 0:
        return np.zeros((8,), dtype=np.float32)
    return s


def _gt_chunk(entry: dict[str, Any]) -> np.ndarray:
    g = np.asarray(entry.get("gt_action_chunk", []), dtype=np.float32)
    return g if g.ndim == 2 else np.zeros((0, 0), dtype=np.float32)


def _camera_path(entry: dict[str, Any], camera: str | None) -> Path | None:
    paths = entry.get("video_paths") or {}
    if not paths:
        return None
    if camera and camera in paths:
        return Path(paths[camera])
    return Path(next(iter(paths.values())))


def _build_history(manifest_by_episode: dict[int, list[dict[str, Any]]],
                   entry: dict[str, Any], history_len: int,
                   action_dim: int) -> np.ndarray:
    """Return (history_len, action_dim) array of GT actions from prior entries
    in the same episode. Older entries pad with zeros if the index is out of range."""
    if history_len <= 0:
        return np.zeros((0, action_dim), dtype=np.float32)
    eps = manifest_by_episode.get(int(entry.get("episode_index", -1)), [])
    eps_sorted = sorted(eps, key=lambda r: int(r.get("frame_index", 0)))
    try:
        cur_idx = next(i for i, r in enumerate(eps_sorted)
                       if r.get("example_id") == entry.get("example_id"))
    except StopIteration:
        cur_idx = -1
    out = np.zeros((history_len, action_dim), dtype=np.float32)
    for k in range(history_len):
        prev_idx = cur_idx - 1 - k
        if prev_idx < 0:
            break
        prev_chunk = _gt_chunk(eps_sorted[prev_idx])
        if prev_chunk.size == 0:
            continue
        # Use the *first* action of the previous chunk as the "executed" action.
        a = prev_chunk[0]
        if a.size >= action_dim:
            out[k] = a[:action_dim]
        else:
            out[k, : a.size] = a
    return out


def _load_target_heatmap(maps_dir: Path, example_id: str,
                         primary_op: str, primary_metric: str) -> np.ndarray | None:
    npz = maps_dir / example_id / "heatmaps.npz"
    if not npz.exists():
        return None
    try:
        with np.load(npz) as z:
            key = f"{primary_op}__{primary_metric}"
            if key not in z.files:
                return None
            arr = z[key].astype(np.float32)
    except Exception as e:
        logger.warning("Failed to load heatmap %s: %s", npz, e)
        return None
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=0)
    return np.nan_to_num(arr, nan=0.0)


def _normalize_target(arr: np.ndarray) -> np.ndarray:
    """Subtract min and L1-normalise so the target sums to 1 (probability mass)."""
    a = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    a = a - a.min()
    s = a.sum()
    return a / s if s > 1e-8 else np.zeros_like(a)


class PerturbMapDataset(Dataset):
    """Yields (image, proprio, lang_emb, history, target, meta) per example."""

    def __init__(self,
                 maps_dir: Path,
                 manifest_path: Path,
                 primary_op: str = "local_mean",
                 primary_metric: str = "action_l2",
                 image_size: tuple[int, int] = (180, 320),
                 history_len: int = 4,
                 action_dim: int = 8,
                 lang_dim: int = 64,
                 camera: str | None = "video.exterior_image_1_left",
                 task_groups: list[str] | None = None,
                 example_ids: list[str] | None = None,
                 normalize_target: bool = True) -> None:
        super().__init__()
        self.maps_dir = Path(maps_dir)
        self.image_size = tuple(image_size)
        self.history_len = int(history_len)
        self.action_dim = int(action_dim)
        self.camera = camera
        self.normalize_target = normalize_target
        self.lang = HashLanguageEmbedder(dim=lang_dim)

        with manifest_path.open("r") as f:
            manifest = json.load(f)
        self._manifest_by_id = {e["example_id"]: e for e in manifest}
        self._manifest_by_episode: dict[int, list[dict[str, Any]]] = {}
        for e in manifest:
            ep = int(e.get("episode_index", -1))
            self._manifest_by_episode.setdefault(ep, []).append(e)

        # Discover usable examples — those that have a heatmap on disk.
        ids = []
        for sub in sorted(self.maps_dir.iterdir()) if self.maps_dir.exists() else []:
            if sub.is_dir() and (sub / "heatmaps.npz").exists():
                ids.append(sub.name)
        if example_ids is not None:
            keep = set(example_ids)
            ids = [i for i in ids if i in keep]
        if task_groups is not None:
            tg_set = set(task_groups)
            ids = [i for i in ids if (self._manifest_by_id.get(i) or {}).get("task_group") in tg_set]

        # Build specs
        self._specs: list[ExampleSpec] = []
        target_shapes: set[tuple[int, int]] = set()
        for ex_id in ids:
            entry = self._manifest_by_id.get(ex_id)
            if entry is None:
                continue
            cam_path = _camera_path(entry, camera)
            if cam_path is None or not cam_path.exists():
                continue
            tgt = _load_target_heatmap(self.maps_dir, ex_id, primary_op, primary_metric)
            if tgt is None:
                continue
            hist = _build_history(self._manifest_by_episode, entry, self.history_len, self.action_dim)
            self._specs.append(ExampleSpec(
                example_id=ex_id,
                task_group=str(entry.get("task_group", "")),
                role=str(entry.get("role", "")),
                episode_index=int(entry.get("episode_index", -1)),
                frame_index=int(entry.get("frame_index", 0)),
                image_path=cam_path,
                proprio=_flatten_state(entry),
                instruction=str(entry.get("instruction", "") or ""),
                prev_action_chunk=hist.astype(np.float32),
                target_heatmap=tgt,
            ))
            target_shapes.add(tgt.shape)

        if len(target_shapes) > 1:
            logger.warning("Heatmaps have multiple shapes: %s — output target_size will use the most common.",
                           target_shapes)
        self._target_shape: tuple[int, int] = (
            max(target_shapes, key=lambda s: list(target_shapes).count(s)) if target_shapes else (0, 0)
        )

        # Cache proprio_dim + lang_dim
        self.proprio_dim = int(self._specs[0].proprio.shape[0]) if self._specs else 0
        self.lang_dim = int(lang_dim)

    @property
    def specs(self) -> list[ExampleSpec]:
        return self._specs

    @property
    def target_shape(self) -> tuple[int, int]:
        return self._target_shape

    def split(self, val_groups: list[str] | None,
              val_fraction: float, seed: int) -> tuple["PerturbMapDataset", "PerturbMapDataset"]:
        """Return (train, val) datasets that share preprocessing config."""
        rng = np.random.default_rng(seed)
        if val_groups:
            train_specs = [s for s in self._specs if s.task_group not in set(val_groups)]
            val_specs = [s for s in self._specs if s.task_group in set(val_groups)]
        else:
            idxs = np.arange(len(self._specs)); rng.shuffle(idxs)
            n_val = max(1, int(round(len(self._specs) * val_fraction)))
            val_idx = set(idxs[:n_val].tolist())
            train_specs = [s for i, s in enumerate(self._specs) if i not in val_idx]
            val_specs = [s for i, s in enumerate(self._specs) if i in val_idx]
        train = self._clone_with(train_specs)
        val = self._clone_with(val_specs)
        return train, val

    def _clone_with(self, new_specs: list[ExampleSpec]) -> "PerturbMapDataset":
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new._specs = list(new_specs)
        return new

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        spec = self._specs[idx]
        # Load image at frame_index (best-effort: use OpenCV)
        try:
            import cv2
            cap = cv2.VideoCapture(str(spec.image_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(spec.frame_index))
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                rgb = np.zeros((*self.image_size, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (rgb.shape[0], rgb.shape[1]) != self.image_size:
                    rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]),
                                     interpolation=cv2.INTER_LINEAR)
        except Exception:
            rgb = np.zeros((*self.image_size, 3), dtype=np.uint8)

        img_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        img_t = (img_t - 0.5) / 0.5                                       # to [-1, 1]
        proprio_t = torch.from_numpy(spec.proprio).float()
        lang_v = self.lang.embed(spec.instruction)
        lang_t = torch.from_numpy(lang_v).float()
        hist_t = torch.from_numpy(spec.prev_action_chunk).float()
        tgt = spec.target_heatmap
        if self.normalize_target:
            tgt = _normalize_target(tgt)
        tgt_t = torch.from_numpy(tgt).float()

        return {
            "image": img_t, "proprio": proprio_t, "lang_emb": lang_t,
            "history": hist_t, "target": tgt_t,
            "example_id": spec.example_id, "task_group": spec.task_group,
            "role": spec.role, "episode_index": spec.episode_index,
            "frame_index": spec.frame_index,
        }


# ============================================================================
# Allocator network
# ============================================================================


class Allocator(nn.Module):
    """Lightweight image-conditioned saliency predictor.

    Architecture
    ------------
    Image branch          : 4-stage 2-D CNN (kernel 5 then 3, stride 2)
                             producing a (B, hidden, H/16, W/16) feature map.
    Conditioning branch   : MLP over the *enabled* subset of {proprio,
                             lang_emb, action history} → (B, hidden) injected
                             as a bias term across the spatial feature.
    Decoder               : conv-block → 1-channel score map → bilinear
                             interpolate to `target_size`.

    Output is in **score-space** (no softmax / no clipping). Use the
    accompanying `predict_normalised` helper to convert to a probability
    map suitable for top-k retention.
    """

    def __init__(self,
                 image_size: tuple[int, int] = (180, 320),
                 proprio_dim: int = 8,
                 lang_emb_dim: int = 64,
                 history_len: int = 4,
                 action_dim: int = 8,
                 target_size: tuple[int, int] = (11, 20),
                 hidden: int = 64,
                 use_lang: bool = True,
                 use_proprio: bool = True,
                 use_history: bool = True) -> None:
        super().__init__()
        self.image_size = tuple(image_size)
        self.target_size = tuple(target_size)
        self.use_lang = bool(use_lang)
        self.use_proprio = bool(use_proprio)
        self.use_history = bool(use_history)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.SiLU(),     # /2
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.SiLU(),    # /4
            nn.Conv2d(32, hidden, kernel_size=3, stride=2, padding=1), nn.SiLU(),# /8
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1), nn.SiLU(),  # /16
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.SiLU(),
        )

        cond_in = 0
        if use_proprio: cond_in += proprio_dim
        if use_lang:    cond_in += lang_emb_dim
        if use_history: cond_in += history_len * action_dim
        self._cond_in = cond_in
        if cond_in > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_in, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.cond_mlp = None

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self,
                image: torch.Tensor,
                proprio: torch.Tensor | None,
                lang_emb: torch.Tensor | None,
                history: torch.Tensor | None) -> torch.Tensor:
        feat = self.encoder(image)                       # (B, hidden, H/16, W/16)
        if self.cond_mlp is not None:
            parts: list[torch.Tensor] = []
            if self.use_proprio and proprio is not None:
                parts.append(proprio.flatten(1).float())
            if self.use_lang and lang_emb is not None:
                parts.append(lang_emb.float())
            if self.use_history and history is not None:
                parts.append(history.flatten(1).float())
            if parts:
                cond_in = torch.cat(parts, dim=-1)
                cond = self.cond_mlp(cond_in)            # (B, hidden)
                feat = feat + cond.view(*cond.shape, 1, 1)
        out = self.decoder(feat)                         # (B, 1, H/16, W/16)
        out = F.interpolate(out, size=self.target_size,
                            mode="bilinear", align_corners=False)
        return out.squeeze(1)                            # (B, target_h, target_w)


# ============================================================================
# Saving predictions in Stage-1 layout
# ============================================================================


def save_predictions_npz(out_dir: Path, example_id: str,
                         predicted_2d: np.ndarray,
                         primary_op: str, primary_metric: str,
                         expand_T: int = 1) -> None:
    """Write a `heatmaps.npz` under `out_dir/<example_id>/` mimicking Stage 1's
    shape. Stage 3 / Stage 4 readers expect `<op>__<metric>` keys + a
    `<op>__valid` mask. Predictions are broadcast over T."""
    sub = Path(out_dir) / example_id
    sub.mkdir(parents=True, exist_ok=True)
    arr = np.nan_to_num(predicted_2d.astype(np.float32), nan=0.0)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr[None, :, :], (max(1, expand_T), *arr.shape)).copy()
    valid = np.ones_like(arr, dtype=np.int8)
    np.savez_compressed(
        sub / "heatmaps.npz",
        **{f"{primary_op}__{primary_metric}": arr,
           f"{primary_op}__valid": valid},
    )


# ============================================================================
# Metrics over predicted vs target maps
# ============================================================================


def _flatten_pos(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr.astype(np.float64), nan=0.0).flatten()
    a = a - a.min()
    return a


def top_k_mask(arr: np.ndarray, k_pct: float) -> np.ndarray:
    flat = _flatten_pos(arr)
    if flat.size == 0:
        return np.zeros_like(arr, dtype=bool)
    n = max(1, int(math.ceil(flat.size * k_pct / 100.0)))
    th = np.partition(flat, -n)[-n]
    return (flat >= th).reshape(arr.shape)


def iou_top_k(a: np.ndarray, b: np.ndarray, k_pct: float) -> float:
    A = top_k_mask(a, k_pct); B = top_k_mask(b, k_pct)
    if not A.any() and not B.any():
        return float("nan")
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter / union) if union > 0 else float("nan")


def recall_top_k(target: np.ndarray, pred: np.ndarray, k_pct: float) -> float:
    T = top_k_mask(target, k_pct); P = top_k_mask(pred, k_pct)
    if T.sum() == 0:
        return float("nan")
    return float(np.logical_and(T, P).sum() / T.sum())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.nan_to_num(a.astype(np.float64), nan=0.0).flatten()
    y = np.nan_to_num(b.astype(np.float64), nan=0.0).flatten()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = np.nan_to_num(a.astype(np.float64), nan=0.0).flatten()
    y = np.nan_to_num(b.astype(np.float64), nan=0.0).flatten()
    if x.size < 4 or x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")
    rx = _rank(x); ry = _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(a.size)
    return ranks


def all_metrics(pred: np.ndarray, target: np.ndarray, k_pcts: Iterable[float]) -> dict[str, float]:
    out: dict[str, float] = {
        "pearson": pearson(pred, target),
        "spearman": spearman(pred, target),
    }
    for k in k_pcts:
        out[f"iou_top_{int(k)}"] = iou_top_k(pred, target, k)
        out[f"recall_top_{int(k)}"] = recall_top_k(target, pred, k)
    return out
