#!/usr/bin/env python3
"""Stage 4 / Server — CALA-WAM-allocation policy server for closed-loop eval.

Wraps `GrootSimPolicy` so that on every inference call the action head's
VAE-encoded conditioning latents (`head.ys`) are *redistributed by importance*
before the DiT denoises them. This is the closed-loop counterpart to
`scripts/stage3/allocation_compute.py` — same operators, same variants,
exposed over the same WebSocket protocol that `eval_utils/run_sim_eval.py`
already speaks (so we drop straight into existing DROID/RoboArena harnesses).

Methods (online importance maps; one forward per inference call):

    none                — pass-through (vanilla DreamZero, no allocation).
    uniform             — flat (every cell equal).
    center              — analytic Gaussian on (0.5, 0.5).
    gripper             — analytic Gaussian on (0.78, 0.5).
    random              — i.i.d. uniform per call (recomputed each step).
    online_attention    — norm of last WAN-DiT block output, captured by hook.
    online_clip         — norm of CLIP image-encoder spatial-token output.
    online_hybrid       — α · clip + (1-α) · attention  (cheap aggregate).
    action_causal_oracle — precomputed Stage-1 importance map loaded from
                           `--importance_map_path` (offline upper bound).

Variants (compression of *non-retained* blocks; same as Stage 3):

    A_hard_retention    — identity inside top-k%, local_mean fill outside.
    B_soft_fidelity     — high (identity) / mid (blur k=3) / low (local_mean) tiers.
    C_temporal_cache    — non-retained replaced by previous-session ys
                          (skips back to A on first call of a session).

Compute knobs:

    --diffusion_steps N — overrides head.num_inference_steps + dit_step_mask.

Diagnostics:

    --save_diagnostics  — writes one JSONL row per inference call with
                          input frame, importance map, decoded future video,
                          predicted action chunk. Used by the rollout figure
                          in `analyze_closed_loop.py`.

Usage example (single-process, single-GPU; matches `serve_dreamzero_wan22.py`):

    python scripts/stage4/server_allocation.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --method online_hybrid --budget 50 --variant A_hard_retention \\
        --port 8000 --save_diagnostics --diagnostics_dir runs/stage4/diag/online_hybrid_b50_A
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("stage4.server")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

import torch.nn.functional as F  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from tianshou.data import Batch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi_client.base_policy import BasePolicy  # noqa: E402
from eval_utils.policy_server import WebsocketPolicyServer, PolicyServerConfig  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
sys.path.insert(0, str(STAGE0_DIR))
from _common import EMBODIMENT_TAG  # noqa: E402


# ============================================================================
# AR_droid observation/action mapping (matches the existing servers)
# ============================================================================

VIDEO_KEY_MAPPING = {
    "observation/exterior_image_0_left": "video.exterior_image_1_left",
    "observation/exterior_image_1_left": "video.exterior_image_2_left",
    "observation/wrist_image_left": "video.wrist_image_left",
}
LANGUAGE_KEY = "annotation.language.action_text"
FRAMES_PER_CHUNK = 4


def _resize_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    import cv2
    if frames.ndim == 3:
        if (frames.shape[0], frames.shape[1]) == (target_h, target_w):
            return frames
        return cv2.resize(frames, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    out = np.stack(
        [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in frames],
        axis=0,
    )
    return out


# ============================================================================
# Importance map providers
# ============================================================================


def _gaussian_2d(shape: tuple[int, int], cy_frac: float, cx_frac: float,
                 sigma_frac: float) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cy = cy_frac * (H - 1); cx = cx_frac * (W - 1)
    sy = max(1.0, sigma_frac * H); sx = max(1.0, sigma_frac * W)
    return np.exp(-((yy - cy) ** 2 / (2 * sy ** 2)
                   + (xx - cx) ** 2 / (2 * sx ** 2))).astype(np.float32)


def _broadcast_3d(arr2: np.ndarray, T: int) -> np.ndarray:
    return np.broadcast_to(arr2[None, :, :], (T, *arr2.shape)).copy()


def _resize_2d(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    import cv2
    Ht, Wt = target_hw
    a = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    return cv2.resize(a, (Wt, Ht), interpolation=cv2.INTER_LINEAR)


def _static_importance(method: str, latent_shape: tuple[int, int, int],
                       rng: np.random.Generator) -> np.ndarray | None:
    T, H, W = latent_shape
    if method == "uniform":
        return np.ones((T, H, W), dtype=np.float32)
    if method == "center":
        return _broadcast_3d(_gaussian_2d((H, W), 0.5, 0.5, 0.18), T)
    if method == "gripper":
        return _broadcast_3d(_gaussian_2d((H, W), 0.78, 0.5, 0.16), T)
    if method == "random":
        return rng.uniform(0, 1, size=(T, H, W)).astype(np.float32)
    return None


def _clip_importance(activation: torch.Tensor | None,
                     latent_shape: tuple[int, int, int]) -> np.ndarray | None:
    if activation is None or not torch.is_tensor(activation) or activation.ndim != 3:
        return None
    T, H, W = latent_shape
    o = activation.float()
    feat = o[0].cpu().numpy()
    norms = np.linalg.norm(feat, axis=-1)
    grid = norms[1:] if norms.size > 1 else norms
    n_grid = int(round(np.sqrt(max(1, grid.size))))
    if n_grid * n_grid != grid.size:
        for h in range(2, int(np.sqrt(grid.size)) + 2):
            if grid.size % h == 0:
                w = grid.size // h
                arr = grid.reshape(h, w)
                lat2 = _resize_2d(arr.astype(np.float32), (H, W))
                return _broadcast_3d(lat2, T)
        return None
    arr = grid.reshape(n_grid, n_grid)
    lat2 = _resize_2d(arr.astype(np.float32), (H, W))
    return _broadcast_3d(lat2, T)


def _dit_importance(activation: torch.Tensor | None,
                    latent_shape: tuple[int, int, int]) -> np.ndarray | None:
    if activation is None or not torch.is_tensor(activation) or activation.ndim != 3:
        return None
    T, H, W = latent_shape
    feat = activation[0].float().cpu().numpy()  # (N, D)
    norms = np.linalg.norm(feat, axis=-1)
    h_d = max(1, H // 2); w_d = max(1, W // 2)
    if T * h_d * w_d == norms.size:
        arr = norms.reshape(T, h_d, w_d)
    else:
        per_t = norms.size // max(1, T)
        side = int(round(np.sqrt(max(1, per_t))))
        if side * side != per_t:
            return None
        arr = norms[:T * side * side].reshape(T, side, side)
    out_resized = np.stack(
        [_resize_2d(arr[t].astype(np.float32), (H, W)) for t in range(T)], axis=0
    )
    return out_resized


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    mn = float(a.min()); mx = float(a.max())
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


# ============================================================================
# Allocation: build perturbed ys from importance + budget + variant
# ============================================================================


def _retention_mask(importance: np.ndarray, budget_pct: float) -> np.ndarray:
    flat = np.nan_to_num(importance.astype(np.float64), nan=0.0).flatten()
    if flat.size == 0:
        return np.zeros_like(importance, dtype=bool)
    n_keep = max(1, int(round(flat.size * budget_pct / 100.0)))
    n_keep = min(n_keep, flat.size)
    thresh = np.partition(flat, -n_keep)[-n_keep]
    return (flat >= thresh).reshape(importance.shape)


def _split_channels(ys: torch.Tensor, ch0: int) -> tuple[torch.Tensor, torch.Tensor]:
    return ys[:, :ch0, :, :, :], ys[:, ch0:, :, :, :]


def _join_channels(mask: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    return torch.cat([mask, lat], dim=1)


def _retained_channel_mean(lat_ch: torch.Tensor, retain_mask: np.ndarray) -> torch.Tensor:
    """Per-channel mean of latents *inside* the retained region (used as fill)."""
    keep = torch.from_numpy(retain_mask.astype(np.float32)).to(lat_ch.device)
    keep_5d = keep[None, None, :, :, :].to(lat_ch.dtype)
    full = lat_ch.float()
    denom = keep.sum().clamp_min(1.0)
    return ((full * keep_5d).sum(dim=(2, 3, 4), keepdim=True) / denom).to(lat_ch.dtype)


def apply_variant_A(ys_base: torch.Tensor, retain: np.ndarray, ch0: int) -> torch.Tensor:
    mask_ch, lat_ch = _split_channels(ys_base, ch0)
    keep = torch.from_numpy(retain.astype(np.float32)).to(lat_ch.device)
    keep5 = keep[None, None, :, :, :].to(lat_ch.dtype)
    fill = _retained_channel_mean(lat_ch, retain)
    new_lat = lat_ch * keep5 + fill * (1.0 - keep5)
    return _join_channels(mask_ch.clone(), new_lat)


def apply_variant_B(ys_base: torch.Tensor, importance: np.ndarray,
                    budget_pct: float, ch0: int) -> torch.Tensor:
    mask_ch, lat_ch = _split_channels(ys_base, ch0)
    flat = np.nan_to_num(importance.astype(np.float64), nan=0.0).flatten()
    if flat.size == 0:
        return ys_base.clone()
    n = flat.size
    n_keep = max(1, int(round(n * budget_pct / 100.0)))
    n_keep = min(n_keep, n)
    thirds = max(1, n_keep // 3)
    high_thresh = np.partition(flat, -thirds)[-thirds]
    mid_thresh = np.partition(flat, -2 * thirds)[-2 * thirds] if 2 * thirds < n else high_thresh
    low_thresh = np.partition(flat, -n_keep)[-n_keep]
    arr = importance.reshape(importance.shape)
    high_mask = (arr >= high_thresh)
    mid_mask = (arr >= mid_thresh) & ~high_mask
    keep_mask = (arr >= low_thresh)
    low_mask = keep_mask & ~high_mask & ~mid_mask
    drop_mask = ~keep_mask
    full = lat_ch.float()
    B, C, T, H, W = full.shape
    weight = torch.ones((C, 1, 1, 3, 3), dtype=full.dtype, device=full.device) / 9.0
    blurred = F.conv3d(full, weight, padding=(0, 1, 1), groups=C)
    fill = _retained_channel_mean(lat_ch, keep_mask).float()
    keep_t = torch.from_numpy(high_mask.astype(np.float32)).to(full.device)[None, None]
    mid_t = torch.from_numpy(mid_mask.astype(np.float32)).to(full.device)[None, None]
    low_t = torch.from_numpy(low_mask.astype(np.float32)).to(full.device)[None, None]
    drop_t = torch.from_numpy(drop_mask.astype(np.float32)).to(full.device)[None, None]
    new_lat = full * keep_t + blurred * mid_t + fill * (low_t + drop_t)
    return _join_channels(mask_ch.clone(), new_lat.to(lat_ch.dtype))


def apply_variant_C(ys_base: torch.Tensor, retain: np.ndarray, ch0: int,
                    prev_ys: torch.Tensor | None) -> torch.Tensor:
    if prev_ys is None:
        return apply_variant_A(ys_base, retain, ch0)
    mask_ch, lat_ch = _split_channels(ys_base, ch0)
    _, prev_lat = _split_channels(prev_ys.to(ys_base.device), ch0)
    keep = torch.from_numpy(retain.astype(np.float32)).to(lat_ch.device)
    keep5 = keep[None, None, :, :, :].to(lat_ch.dtype)
    new_lat = lat_ch * keep5 + prev_lat.to(lat_ch.dtype) * (1.0 - keep5)
    return _join_channels(mask_ch.clone(), new_lat)


def apply_allocation(ys_base: torch.Tensor, importance: np.ndarray | None,
                     budget_pct: float, variant: str, ch0: int,
                     prev_ys: torch.Tensor | None) -> torch.Tensor:
    if budget_pct >= 100.0 - 1e-6 or importance is None:
        return ys_base
    retain = _retention_mask(importance, budget_pct)
    if variant == "A_hard_retention":
        return apply_variant_A(ys_base, retain, ch0)
    if variant == "B_soft_fidelity":
        return apply_variant_B(ys_base, importance, budget_pct, ch0)
    if variant == "C_temporal_cache":
        return apply_variant_C(ys_base, retain, ch0, prev_ys=prev_ys)
    return apply_variant_A(ys_base, retain, ch0)


# ============================================================================
# Activation hooks for online importance computation
# ============================================================================


@contextmanager
def _capture_hooks(head: Any):
    captured: dict[str, torch.Tensor | None] = {"image_encoder": None, "dit_last_block": None}
    handles: list[Any] = []

    def _make_hook(name: str):
        def hook(_mod, _inputs, outputs):
            if captured[name] is not None:
                return
            out = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
            if torch.is_tensor(out):
                captured[name] = out.detach()
        return hook

    try:
        ie = getattr(head, "image_encoder", None)
        if ie is not None:
            handles.append(ie.register_forward_hook(_make_hook("image_encoder")))
        blocks = getattr(getattr(head, "model", None), "blocks", None)
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
            try:
                handles.append(blocks[len(blocks) - 1].register_forward_hook(_make_hook("dit_last_block")))
            except Exception:
                pass
        yield captured
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


@contextmanager
def _patched_encode_image(head: Any, mutator):
    """Wrap head.encode_image so its returned ys is replaced by `mutator(clip, ys, new_image)`."""
    original = head.encode_image

    def wrapped(image, num_frames, height, width):
        clip, ys, new_image = original(image, num_frames, height, width)
        try:
            new_ys = mutator(clip, ys, new_image)
            if new_ys is not None and torch.is_tensor(new_ys) and new_ys.shape == ys.shape:
                ys = new_ys
        except Exception as e:
            logger.warning("allocation mutator failed: %s", e)
        return clip, ys, new_image
    head.encode_image = wrapped
    try:
        yield
    finally:
        head.encode_image = original


# ============================================================================
# AllocationPolicy
# ============================================================================


class AllocationPolicy(BasePolicy):
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        method: str,
        budget_pct: float,
        variant: str,
        diffusion_steps: int | None,
        importance_map_path: str | None,
        clip_attention_alpha: float,
        save_diagnostics: bool,
        diagnostics_dir: str,
        image_height: int,
        image_width: int,
        seed: int,
    ) -> None:
        super().__init__()
        self._policy = groot_policy
        self._head = groot_policy.trained_model.action_head
        self._method = method
        self._budget = float(budget_pct)
        self._variant = variant
        self._clip_alpha = float(clip_attention_alpha)
        self._image_height = image_height
        self._image_width = image_width
        self._save_diagnostics = save_diagnostics
        self._diagnostics_dir = Path(diagnostics_dir)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._frame_buffers = {k: [] for k in VIDEO_KEY_MAPPING.values()}
        self._is_first_call = True
        self._current_session_id: str | None = None
        self._cached_importance: np.ndarray | None = None
        self._prev_ys: torch.Tensor | None = None
        self._call_index_in_session: int = 0

        # Optional precomputed map for `action_causal_oracle`
        self._precomputed_map: np.ndarray | None = None
        if importance_map_path:
            try:
                self._precomputed_map = np.load(importance_map_path).astype(np.float32)
            except Exception as e:
                logger.warning("Failed to load importance_map_path=%s: %s", importance_map_path, e)

        # Diffusion-step override
        if diffusion_steps and diffusion_steps > 0:
            try:
                self._head.num_inference_steps = int(diffusion_steps)
                self._head.dit_step_mask = [True] * int(diffusion_steps)
                logger.info("Diffusion steps overridden -> %d", diffusion_steps)
            except Exception as e:
                logger.warning("Diffusion-step override failed: %s", e)

        if self._save_diagnostics:
            self._diagnostics_dir.mkdir(parents=True, exist_ok=True)

        self._z_channel_start: int = 4

    # ------------------------------------------------------------------
    # Static / cached importance helpers
    # ------------------------------------------------------------------

    def _get_importance(self, latent_shape: tuple[int, int, int],
                        captured: dict[str, torch.Tensor | None] | None) -> np.ndarray | None:
        if self._method == "none":
            return None
        if self._method in ("uniform", "center", "gripper"):
            if self._cached_importance is None:
                self._cached_importance = _static_importance(self._method, latent_shape, self._rng)
            return self._cached_importance
        if self._method == "random":
            return _static_importance("random", latent_shape, self._rng)
        if self._method == "action_causal_oracle":
            if self._precomputed_map is None:
                return None
            arr = self._precomputed_map
            if arr.shape != latent_shape:
                arr2 = arr if arr.ndim == 2 else np.nanmean(arr, axis=0)
                arr2 = _resize_2d(arr2.astype(np.float32),
                                  (latent_shape[1], latent_shape[2]))
                arr = _broadcast_3d(arr2, latent_shape[0])
            return arr.astype(np.float32)
        # Online methods need previous-step activations
        if captured is None:
            return None
        clip_imp = _clip_importance(captured.get("image_encoder"), latent_shape)
        dit_imp = _dit_importance(captured.get("dit_last_block"), latent_shape)
        if self._method == "online_clip":
            return clip_imp
        if self._method == "online_attention":
            return dit_imp
        if self._method == "online_hybrid":
            n_clip = _normalize(clip_imp) if clip_imp is not None else None
            n_dit = _normalize(dit_imp) if dit_imp is not None else None
            if n_clip is None and n_dit is None:
                return None
            if n_clip is None:
                return n_dit
            if n_dit is None:
                return n_clip
            return (self._clip_alpha * n_clip + (1.0 - self._clip_alpha) * n_dit).astype(np.float32)
        return None

    # ------------------------------------------------------------------
    # Observation conversion (mirrors serve_dreamzero_wan22)
    # ------------------------------------------------------------------

    def _convert_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        for arena_key, model_key in VIDEO_KEY_MAPPING.items():
            if arena_key in obs:
                data = np.asarray(obs[arena_key])
                data = _resize_frames(data, self._image_height, self._image_width)
                if data.ndim == 4:
                    self._frame_buffers[model_key].extend(list(data))
                else:
                    self._frame_buffers[model_key].append(data)
        num_frames = 1 if self._is_first_call else FRAMES_PER_CHUNK
        converted: dict[str, Any] = {}
        for model_key, buf in self._frame_buffers.items():
            if not buf:
                continue
            if len(buf) >= num_frames:
                frames = buf[-num_frames:]
            else:
                frames = buf.copy()
                while len(frames) < num_frames:
                    frames.insert(0, buf[0])
            converted[model_key] = np.stack(frames, axis=0)

        joint = np.asarray(obs.get("observation/joint_position", np.zeros(7))).reshape(1, -1)
        converted["state.joint_position"] = joint.astype(np.float64)
        gripper = np.asarray(obs.get("observation/gripper_position", np.zeros(1))).reshape(1, -1)
        converted["state.gripper_position"] = gripper.astype(np.float64)
        converted[LANGUAGE_KEY] = obs.get("prompt", "") or ""
        return converted

    @staticmethod
    def _convert_action(action_dict: dict[str, Any]) -> np.ndarray:
        joint = None; gripper = None
        for k, v in action_dict.items():
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            if "joint_position" in k and "gripper" not in k:
                joint = v
            elif "gripper" in k:
                gripper = v
        if joint is None:
            return np.zeros((1, 8), dtype=np.float32)
        if joint.ndim == 1:
            joint = joint.reshape(1, -1)
        N = joint.shape[0]
        if gripper is None:
            gripper = np.zeros((N, 1), dtype=np.float32)
        elif gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        if gripper.shape[-1] > 1:
            gripper = gripper[..., :1]
        return np.concatenate([joint, gripper], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _maybe_save_diag(self, step: int, obs: dict[str, Any], action: np.ndarray,
                        importance: np.ndarray | None, latent_shape: tuple[int, int, int],
                        elapsed_s: float) -> None:
        if not self._save_diagnostics:
            return
        try:
            diag = {
                "session_id": self._current_session_id,
                "step": int(step),
                "method": self._method,
                "budget_pct": self._budget,
                "variant": self._variant,
                "elapsed_s": float(elapsed_s),
                "action_shape": list(action.shape),
                "action_first": action[0].tolist() if action.size else [],
                "latent_shape": list(latent_shape),
                "has_importance": importance is not None,
                "prompt": obs.get("prompt", "") or "",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            if importance is not None:
                diag["importance_2d"] = np.nanmean(importance, axis=0).tolist()
            session_dir = self._diagnostics_dir / (self._current_session_id or "default")
            session_dir.mkdir(parents=True, exist_ok=True)
            with (session_dir / "diag.jsonl").open("a") as f:
                f.write(json.dumps(diag) + "\n")
        except Exception as e:
            logger.debug("diagnostics save failed: %s", e)

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        # Session change → reset
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                self.reset({})
            self._current_session_id = session_id

        converted = self._convert_observation(obs)

        # Build allocation mutator using importance from previous step (cached) or static.
        latent_shape_holder = {"shape": None}
        importance_used: np.ndarray | None = None

        def mutator(clip_feas, ys, new_image):
            latent_shape_holder["shape"] = (int(ys.shape[2]), int(ys.shape[3]), int(ys.shape[4]))
            self._z_channel_start = 4 if int(ys.shape[1]) > 4 else 0
            importance = self._get_importance(latent_shape_holder["shape"], None)
            if importance is None and self._method.startswith("online_"):
                importance = self._cached_importance
            nonlocal importance_used
            importance_used = importance
            return apply_allocation(
                ys, importance, self._budget, self._variant,
                self._z_channel_start, self._prev_ys,
            )

        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

        t0 = time.perf_counter()
        with _capture_hooks(self._head) as captured, \
             _patched_encode_image(self._head, mutator), \
             torch.inference_mode():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(
                Batch(obs=converted)
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Update cached importance from this step's activations (for the next call).
        if self._method.startswith("online_") and latent_shape_holder["shape"] is not None:
            new_imp = self._get_importance(latent_shape_holder["shape"], captured)
            if new_imp is not None:
                self._cached_importance = new_imp

        # Snapshot current ys for variant C of the *next* call
        if self._variant == "C_temporal_cache":
            try:
                self._prev_ys = self._head.ys.detach().clone()
            except Exception:
                self._prev_ys = None

        # Convert action
        action_dict = {k: getattr(result_batch.act, k) for k in dir(result_batch.act)
                       if k.startswith("action.")}
        action = self._convert_action(action_dict)
        if self._is_first_call:
            self._is_first_call = False

        self._maybe_save_diag(
            self._call_index_in_session, obs, action,
            importance_used, latent_shape_holder["shape"] or (0, 0, 0), elapsed,
        )
        self._call_index_in_session += 1
        return action

    def reset(self, _reset_info: dict[str, Any]) -> None:
        # Clear causal-state in the action head + our buffers
        for attr in ("current_start_frame", "language", "clip_feas", "ys"):
            if hasattr(self._head, attr):
                if attr == "current_start_frame":
                    setattr(self._head, attr, 0)
                else:
                    setattr(self._head, attr, None)
        for k in list(self._frame_buffers.keys()):
            self._frame_buffers[k] = []
        self._is_first_call = True
        self._call_index_in_session = 0
        # Keep cached importance across episodes for sticky priors except 'random'
        if self._method == "random":
            self._cached_importance = None
        # variant C cache only valid within an episode
        self._prev_ys = None


# ============================================================================
# Driver
# ============================================================================


def _maybe_init_distributed() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                            rank=0, world_size=1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--method", default="online_hybrid",
                   choices=["none", "uniform", "center", "gripper", "random",
                            "online_clip", "online_attention", "online_hybrid",
                            "action_causal_oracle"])
    p.add_argument("--budget", type=float, default=50.0)
    p.add_argument("--variant", default="A_hard_retention",
                   choices=["A_hard_retention", "B_soft_fidelity", "C_temporal_cache"])
    p.add_argument("--diffusion_steps", type=int, default=None)
    p.add_argument("--importance_map_path", default=None)
    p.add_argument("--clip_attention_alpha", type=float, default=0.5,
                   help="Mixing weight for online_hybrid (clip vs attention).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--image_height", type=int, default=180)
    p.add_argument("--image_width", type=int, default=320)
    p.add_argument("--save_diagnostics", action="store_true")
    p.add_argument("--diagnostics_dir", default="./runs/stage4/diag")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        force=True)

    _maybe_init_distributed()
    if torch.cuda.is_available():
        init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    logger.info("Loading model from %s ...", args.checkpoint)
    groot_policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        model_path=args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logger.info("Model loaded.")

    policy = AllocationPolicy(
        groot_policy=groot_policy,
        method=args.method,
        budget_pct=args.budget,
        variant=args.variant,
        diffusion_steps=args.diffusion_steps,
        importance_map_path=args.importance_map_path,
        clip_attention_alpha=args.clip_attention_alpha,
        save_diagnostics=args.save_diagnostics,
        diagnostics_dir=args.diagnostics_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        seed=args.seed,
    )

    server_config = PolicyServerConfig(
        image_resolution=(args.image_height, args.image_width),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )
    logger.info("Server starting on %s:%d  (method=%s, budget=%.1f, variant=%s, diff_steps=%s)",
                args.host, args.port, args.method, args.budget, args.variant,
                args.diffusion_steps)
    server = WebsocketPolicyServer(
        policy=policy,
        server_config=server_config,
        host=args.host,
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
