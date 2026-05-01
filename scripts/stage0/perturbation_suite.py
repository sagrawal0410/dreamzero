#!/usr/bin/env python3
"""Stage 0 / E0.6 — Perturbation Operator Sanity Suite for DreamZero-DROID.

Decides which perturbation operator(s) to use for the Stage-1 causal-importance
maps. Implements a deliberately redundant suite spanning:

    A. Controls           — identity, copy-and-reinsert, tiny epsilon noise.
    B. Destructive        — zero mask, sign flip, magnitude scale.
    C. In-distribution    — dataset / frame / local-neighborhood mean.
    D. Noise              — matched-std Gaussian, matched-norm Gaussian,
                             channel dropout.
    E. Compression-like   — average pooling, spatial blur (low-pass).
    F. Temporal / cache   — previous-frame cache (lag 1, 2, 4).
    G. Spatial specificity — nearby-region swap.

Each operator is applied to a region of the action head's VAE-encoded
conditioning latents (`head.ys`) — channels 4: only, since channels 0:4 are a
binary frame-validity mask. Re-injection works by monkey-patching
`head.encode_image` to return the (cached_clip_feas, perturbed_ys,
cached_new_image) tuple while everything else in the forward pass stays bit-
identical to the baseline. Same `torch.manual_seed` for baseline and perturbed
(paired-seed protocol) so deltas reflect the perturbation, not sampling noise.

Per (example, operator, strength, region) we record:

    delta_action_l2 / cosine / first / near / far / gripper
    delta_video_pred_l2
    latent_perturbation_norm + relative
    latent_zscore (OOD diagnostic)
    nan_rate / valid
    normalized_effect = delta_action_l2 / (sigma_seed + eps)   if --noise_floor

Outputs (under <output_dir>):

    per_run.csv                       — long-form table, one row per config
    summary_per_operator.json         — aggregated stats and ranking
    operator_ranking.csv              — paper-ready ranking table
    plots/
        plot_strength_sweep__<op>.png/pdf      — Δa vs strength per operator
        plot_operator_heatmap.png/pdf          — operator × region Δa heatmap
        plot_operator_ranking.png/pdf          — bar of validity / locality / sensitivity
        plot_validity.png/pdf                  — NaN / failure rate per operator
    decoded_grid/<example>/<op>__<strength>__<region>.png  — visual sanity
    combined_report.md                — paper-appendix-ready summary + Stage-1 recommendation

Example:

    python scripts/stage0/perturbation_suite.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_examples 4 \\
        --noise_floor runs/stage0_stability/noise_floor.json \\
        --output_dir runs/stage0_perturb
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage0.perturbation")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

import torch.nn.functional as F  # noqa: E402
from tianshou.data import Batch  # noqa: E402

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

# Local helpers
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import (  # noqa: E402
    EMBODIMENT_TAG,
    build_obs,
    configure_neurips_matplotlib,
    flatten_action_dict,
    init_dist_single_process,
    read_frame,
    reset_causal_state,
)


# ============================================================================
# Per-example baseline capture & perturbed re-run
# ============================================================================


@dataclass
class BaselineCache:
    obs: dict[str, Any]
    seed: int
    ys: torch.Tensor                   # (B, 4+z, T_lat, H_lat, W_lat)
    clip_feas: torch.Tensor
    new_image: torch.Tensor | Any
    action_chunk: np.ndarray
    actions_per_key: dict[str, np.ndarray]
    video_pred: torch.Tensor | None
    elapsed_s: float
    z_channel_start: int = 4


@contextmanager
def _patched_encode_image(head: Any, capturing: bool, cache: dict[str, Any] | None,
                          replacement_ys: torch.Tensor | None = None):
    """Monkey-patch head.encode_image either to capture or to return cached/perturbed.

    capturing=True   : record clip_feas/ys/new_image into `cache`.
    capturing=False  : return (cached_clip_feas, replacement_ys, cached_new_image).
    """
    original = head.encode_image
    if capturing:
        def wrapped(image, num_frames, height, width):
            clip, ys, new_image = original(image, num_frames, height, width)
            if cache is not None:
                cache["clip_feas"] = clip.detach().clone()
                cache["ys"] = ys.detach().clone()
                cache["new_image"] = (
                    new_image.detach().clone() if torch.is_tensor(new_image) else new_image
                )
            return clip, ys, new_image
        head.encode_image = wrapped
    else:
        if cache is None or replacement_ys is None:
            raise ValueError("Replacement mode requires cache and replacement_ys.")
        def wrapped(image, num_frames, height, width):
            ni = cache["new_image"]
            ni_ret = ni.clone() if torch.is_tensor(ni) else ni
            return cache["clip_feas"].clone(), replacement_ys, ni_ret
        head.encode_image = wrapped
    try:
        yield
    finally:
        head.encode_image = original


def _run_with_seed(policy: GrootSimPolicy, obs: dict[str, Any], seed: int):
    reset_causal_state(policy)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    actions: dict[str, np.ndarray] = {}
    for k in dir(result.act):
        if k.startswith("action."):
            v = getattr(result.act, k)
            if torch.is_tensor(v):
                v = v.detach().cpu().float().numpy()
            actions[k] = np.asarray(v)
    chunk = flatten_action_dict(actions)
    if torch.is_tensor(video_pred):
        video_pred = video_pred.detach().to("cpu")
    return chunk, actions, video_pred, elapsed


def capture_baseline(policy: GrootSimPolicy, obs: dict[str, Any], seed: int) -> BaselineCache:
    head = policy.trained_model.action_head
    cache: dict[str, Any] = {}
    with _patched_encode_image(head, capturing=True, cache=cache):
        chunk, actions, video_pred, elapsed = _run_with_seed(policy, obs, seed)
    z_channel_start = 4 if cache.get("ys") is not None and cache["ys"].shape[1] > 4 else 0
    return BaselineCache(
        obs=obs,
        seed=seed,
        ys=cache["ys"],
        clip_feas=cache["clip_feas"],
        new_image=cache.get("new_image"),
        action_chunk=chunk,
        actions_per_key=actions,
        video_pred=video_pred,
        elapsed_s=elapsed,
        z_channel_start=z_channel_start,
    )


def run_perturbed(policy: GrootSimPolicy,
                  baseline: BaselineCache,
                  ys_perturbed: torch.Tensor) -> tuple[np.ndarray, dict[str, np.ndarray], torch.Tensor | None, float]:
    head = policy.trained_model.action_head
    cache = {"clip_feas": baseline.clip_feas, "ys": baseline.ys, "new_image": baseline.new_image}
    with _patched_encode_image(head, capturing=False, cache=cache, replacement_ys=ys_perturbed):
        chunk, actions, video_pred, elapsed = _run_with_seed(policy, baseline.obs, baseline.seed)
    return chunk, actions, video_pred, elapsed


# ============================================================================
# Region masks (in latent T × H × W)
# ============================================================================


def _region_slices(ys_shape: tuple[int, ...], name: str,
                   rng: np.random.Generator) -> tuple[slice, slice, slice]:
    """Return (slice_t, slice_h, slice_w) for the named region in latent space."""
    if len(ys_shape) != 5:
        raise ValueError(f"ys is expected to be 5D (B,C,T,H,W); got shape {ys_shape}")
    _, _, T, H, W = ys_shape
    if name == "whole_grid":
        return slice(None), slice(None), slice(None)
    if name == "center":
        return slice(None), slice(max(1, H // 4), max(2, 3 * H // 4)), slice(max(1, W // 4), max(2, 3 * W // 4))
    if name == "top_left":
        return slice(None), slice(0, max(1, H // 3)), slice(0, max(1, W // 3))
    if name == "bottom_right":
        return slice(None), slice(max(1, 2 * H // 3), H), slice(max(1, 2 * W // 3), W)
    if name == "bottom_center":
        return slice(None), slice(max(1, 2 * H // 3), H), slice(max(1, W // 3), max(2, 2 * W // 3))
    if name == "random_block":
        bh = max(1, H // 4); bw = max(1, W // 4)
        h0 = int(rng.integers(0, max(1, H - bh + 1)))
        w0 = int(rng.integers(0, max(1, W - bw + 1)))
        return slice(None), slice(h0, h0 + bh), slice(w0, w0 + bw)
    raise ValueError(f"Unknown region: {name}")


REGION_MEANING = {
    "whole_grid": "entire latent grid",
    "center": "central 50% — proxy for object/contact area",
    "top_left": "top-left corner — proxy for far background",
    "bottom_right": "bottom-right corner — proxy for far background",
    "bottom_center": "bottom-center — proxy for gripper / end-effector",
    "random_block": "uniformly random spatial block",
}


# ============================================================================
# Operator implementations
# ============================================================================
#
# Every operator receives the FULL ys tensor and the region slices, then
# returns a NEW ys tensor with the region replaced. Channels 0:z_channel_start
# (the validity mask) are NEVER perturbed — they are copied from the original
# tensor. This keeps the masked-conditioning semantics intact.
# ----------------------------------------------------------------------------


def _split_channels(ys: torch.Tensor, ch0: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mask_channels, latent_channels)."""
    return ys[:, :ch0, :, :, :], ys[:, ch0:, :, :, :]


def _join_channels(mask_ch: torch.Tensor, lat_ch: torch.Tensor) -> torch.Tensor:
    return torch.cat([mask_ch, lat_ch], dim=1)


def _apply_to_block(ys: torch.Tensor,
                    region: tuple[slice, slice, slice],
                    new_block_lat: torch.Tensor,
                    ch0: int,
                    strength: float) -> torch.Tensor:
    """Replace ys[..., region] latent channels with (1-α) z + α z_tilde."""
    mask_ch, lat_ch = _split_channels(ys, ch0)
    sl_t, sl_h, sl_w = region
    block = lat_ch[:, :, sl_t, sl_h, sl_w]
    new_block = (1.0 - strength) * block + strength * new_block_lat.to(block.dtype)
    out_lat = lat_ch.clone()
    out_lat[:, :, sl_t, sl_h, sl_w] = new_block
    return _join_channels(mask_ch.clone(), out_lat)


# ----- A. Controls ----------------------------------------------------------


def op_identity(ys, region, ch0, strength=0.0, **_kw):
    return ys.clone()


def op_copy_reinsert(ys, region, ch0, strength=1.0, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    return _apply_to_block(ys, region, block, ch0, strength=1.0)


def op_eps_noise(ys, region, ch0, strength=1.0, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    sigma = 1e-3 * float(lat_ch.float().std().clamp_min(1e-8).item())
    new_block = block + torch.randn_like(block) * sigma
    return _apply_to_block(ys, region, new_block, ch0, strength=1.0)


# ----- B. Destructive -------------------------------------------------------


def op_zero_mask(ys, region, ch0, strength=1.0, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    return _apply_to_block(ys, region, torch.zeros_like(block), ch0, strength=strength)


def op_sign_flip(ys, region, ch0, strength=1.0, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    return _apply_to_block(ys, region, -block, ch0, strength=strength)


def op_magnitude_scale(ys, region, ch0, strength=1.0, scale=1.0, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    return _apply_to_block(ys, region, block * float(scale), ch0, strength=1.0)


# ----- C. In-distribution ---------------------------------------------------


def op_dataset_mean(ys, region, ch0, strength=1.0, dataset_mean_lat=None, **_kw):
    if dataset_mean_lat is None:
        return op_identity(ys, region, ch0)
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    target = dataset_mean_lat.to(block.device, dtype=block.dtype).expand_as(block)
    return _apply_to_block(ys, region, target, ch0, strength=strength)


def op_frame_mean(ys, region, ch0, strength=1.0, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    # Mean over T, H, W per batch, per channel
    target = lat_ch.mean(dim=(2, 3, 4), keepdim=True).expand_as(block)
    return _apply_to_block(ys, region, target, ch0, strength=strength)


def op_local_mean(ys, region, ch0, strength=1.0, **_kw):
    """Replace block with mean of latents *outside* the region (whole-image local mean)."""
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    # Build mask 1 outside, 0 inside the region; compute weighted mean
    full = lat_ch.float()
    outside_mask = torch.ones_like(full)
    outside_mask[:, :, region[0], region[1], region[2]] = 0.0
    denom = outside_mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
    outside_mean = (full * outside_mask).sum(dim=(2, 3, 4), keepdim=True) / denom
    target = outside_mean.to(block.dtype).expand_as(block)
    return _apply_to_block(ys, region, target, ch0, strength=strength)


# ----- D. Noise -------------------------------------------------------------


def op_gaussian_noise(ys, region, ch0, strength=0.1, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    std_ref = float(lat_ch.float().std().clamp_min(1e-8).item())
    sigma = float(strength) * std_ref
    new_block = block + torch.randn_like(block) * sigma
    return _apply_to_block(ys, region, new_block, ch0, strength=1.0)


def op_matched_norm_noise(ys, region, ch0, strength=0.1, **_kw):
    """Inject Gaussian noise scaled to a target ratio of the block norm."""
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    block_norm = float(block.float().norm().clamp_min(1e-8).item())
    eps = torch.randn_like(block)
    eps_norm = float(eps.float().norm().clamp_min(1e-8).item())
    eps = eps * (block_norm / eps_norm) * float(strength)
    new_block = block + eps
    return _apply_to_block(ys, region, new_block, ch0, strength=1.0)


def op_channel_dropout(ys, region, ch0, strength=0.5, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    C = block.shape[1]
    n_drop = max(1, int(round(C * float(strength))))
    perm = torch.randperm(C, device=block.device)
    drop_idx = perm[:n_drop]
    new_block = block.clone()
    new_block[:, drop_idx] = 0.0
    return _apply_to_block(ys, region, new_block, ch0, strength=1.0)


# ----- E. Compression-like --------------------------------------------------


def op_avg_pool(ys, region, ch0, strength=1.0, factor=2, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    B, C, T, H, W = block.shape
    if H < 2 or W < 2 or factor < 2:
        return op_identity(ys, region, ch0)
    f = int(factor)
    fh = max(1, min(f, H))
    fw = max(1, min(f, W))
    # Average over fh × fw windows then upsample with nearest to original size
    pooled = F.avg_pool3d(block.float(), kernel_size=(1, fh, fw), stride=(1, fh, fw))
    pooled_up = F.interpolate(pooled, size=(T, H, W), mode="nearest")
    return _apply_to_block(ys, region, pooled_up.to(block.dtype), ch0, strength=strength)


def op_blur(ys, region, ch0, strength=1.0, kernel_size=3, **_kw):
    _, lat_ch = _split_channels(ys, ch0)
    block = lat_ch[:, :, region[0], region[1], region[2]]
    B, C, T, H, W = block.shape
    k = int(kernel_size) | 1  # force odd
    if H < 3 or W < 3:
        return op_identity(ys, region, ch0)
    pad = k // 2
    weight = torch.ones((C, 1, 1, k, k), dtype=block.dtype, device=block.device) / (k * k)
    blurred = F.conv3d(block, weight, padding=(0, pad, pad), groups=C)
    return _apply_to_block(ys, region, blurred, ch0, strength=strength)


# ----- F. Temporal / cache --------------------------------------------------


def op_prev_frame_cache(ys, region, ch0, strength=1.0, prev_ys=None, **_kw):
    """Replace block with the same-coordinate block from a previous-frame ys."""
    if prev_ys is None:
        return op_identity(ys, region, ch0)
    _, lat_ch_p = _split_channels(prev_ys, ch0)
    target = lat_ch_p[:, :, region[0], region[1], region[2]]
    return _apply_to_block(ys, region, target, ch0, strength=strength)


# ----- G. Spatial specificity ----------------------------------------------


def op_nearby_swap(ys, region, ch0, strength=1.0, rng=None, **_kw):
    """Swap region with a random nearby block of the same size."""
    _, lat_ch = _split_channels(ys, ch0)
    sl_t, sl_h, sl_w = region
    bh = sl_h.stop - sl_h.start
    bw = sl_w.stop - sl_w.start
    H, W = lat_ch.shape[-2], lat_ch.shape[-1]
    rng = rng or np.random.default_rng(0)
    # Look for a non-overlapping nearby block within ±2 block sizes
    for _ in range(20):
        h0 = int(rng.integers(max(0, sl_h.start - 2 * bh), min(H - bh, sl_h.stop + 2 * bh) + 1))
        w0 = int(rng.integers(max(0, sl_w.start - 2 * bw), min(W - bw, sl_w.stop + 2 * bw) + 1))
        if abs(h0 - sl_h.start) >= bh or abs(w0 - sl_w.start) >= bw:
            break
    target = lat_ch[:, :, sl_t, h0:h0 + bh, w0:w0 + bw]
    if target.shape != lat_ch[:, :, region[0], region[1], region[2]].shape:
        return op_identity(ys, region, ch0)
    return _apply_to_block(ys, region, target, ch0, strength=strength)


# ============================================================================
# Operator registry (name, fn, strengths, category)
# ============================================================================


@dataclass
class OperatorSpec:
    name: str
    fn: Callable[..., torch.Tensor]
    strengths: list[float]                 # sweep parameter (interpretation per-op)
    category: str
    needs: list[str] = field(default_factory=list)   # extra inputs (dataset_mean_lat, prev_ys)
    extras: dict[str, Any] = field(default_factory=dict)


OPERATORS: list[OperatorSpec] = [
    OperatorSpec("identity",          op_identity,          [0.0],            "control"),
    OperatorSpec("copy_reinsert",     op_copy_reinsert,     [1.0],            "control"),
    OperatorSpec("eps_noise",         op_eps_noise,         [1.0],            "control"),
    OperatorSpec("zero_mask",         op_zero_mask,         [0.25, 0.5, 1.0], "destructive"),
    OperatorSpec("sign_flip",         op_sign_flip,         [0.5, 1.0],       "destructive"),
    OperatorSpec("magnitude_scale_0", op_magnitude_scale,   [1.0],            "destructive", extras={"scale": 0.0}),
    OperatorSpec("magnitude_scale_2", op_magnitude_scale,   [1.0],            "destructive", extras={"scale": 2.0}),
    OperatorSpec("dataset_mean",      op_dataset_mean,      [0.5, 1.0],       "in_distribution",
                 needs=["dataset_mean_lat"]),
    OperatorSpec("frame_mean",        op_frame_mean,        [0.5, 1.0],       "in_distribution"),
    OperatorSpec("local_mean",        op_local_mean,        [0.5, 1.0],       "in_distribution"),
    OperatorSpec("gaussian_noise",    op_gaussian_noise,    [0.05, 0.25, 1.0], "noise"),
    OperatorSpec("matched_norm_noise", op_matched_norm_noise, [0.1, 0.5],     "noise"),
    OperatorSpec("channel_dropout",   op_channel_dropout,   [0.25, 0.5, 0.75], "noise"),
    OperatorSpec("avg_pool_2",        op_avg_pool,          [1.0],            "compression", extras={"factor": 2}),
    OperatorSpec("avg_pool_4",        op_avg_pool,          [1.0],            "compression", extras={"factor": 4}),
    OperatorSpec("blur_k3",           op_blur,              [1.0],            "compression", extras={"kernel_size": 3}),
    OperatorSpec("blur_k5",           op_blur,              [1.0],            "compression", extras={"kernel_size": 5}),
    OperatorSpec("nearby_swap",       op_nearby_swap,       [1.0],            "spatial"),
    OperatorSpec("prev_frame_cache_l1", op_prev_frame_cache, [1.0],           "temporal", needs=["prev_ys_l1"]),
    OperatorSpec("prev_frame_cache_l2", op_prev_frame_cache, [1.0],           "temporal", needs=["prev_ys_l2"]),
    OperatorSpec("prev_frame_cache_l4", op_prev_frame_cache, [1.0],           "temporal", needs=["prev_ys_l4"]),
]


# ============================================================================
# Metrics
# ============================================================================


def _safe_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size and b.size else float("nan")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64); bf = b.flatten().astype(np.float64)
    na = np.linalg.norm(af); nb = np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(af, bf) / (na * nb))


def _action_metrics(base_chunk: np.ndarray, pert_chunk: np.ndarray) -> dict[str, float]:
    if base_chunk.shape != pert_chunk.shape or base_chunk.size == 0:
        return {"delta_l2": float("nan"), "delta_cosine": float("nan"),
                "delta_first": float("nan"), "delta_near": float("nan"),
                "delta_far": float("nan"), "delta_gripper": float("nan"),
                "shape_match": 0}
    H, A = base_chunk.shape
    third = max(1, H // 3)
    out = {
        "delta_l2": _safe_l2(base_chunk, pert_chunk),
        "delta_cosine": _cosine(base_chunk, pert_chunk),
        "delta_first": _safe_l2(base_chunk[:1], pert_chunk[:1]),
        "delta_near": _safe_l2(base_chunk[:third], pert_chunk[:third]),
        "delta_far": _safe_l2(base_chunk[-third:], pert_chunk[-third:]),
        "delta_gripper": _safe_l2(base_chunk[:, -1:], pert_chunk[:, -1:]),
        "shape_match": 1,
    }
    return out


def _video_metrics(vp_base: torch.Tensor | None, vp_pert: torch.Tensor | None) -> float:
    if vp_base is None or vp_pert is None:
        return float("nan")
    if vp_base.shape != vp_pert.shape:
        return float("nan")
    return float(torch.sqrt(((vp_base.float() - vp_pert.float()) ** 2).mean()).item())


def _latent_metrics(ys_base: torch.Tensor, ys_pert: torch.Tensor,
                    region: tuple[slice, slice, slice], ch0: int) -> dict[str, float]:
    diff = (ys_pert - ys_base).float()
    sl_t, sl_h, sl_w = region
    block_diff = diff[:, ch0:, sl_t, sl_h, sl_w]
    block_orig = ys_base[:, ch0:, sl_t, sl_h, sl_w].float()
    block_pert = ys_pert[:, ch0:, sl_t, sl_h, sl_w].float()
    base_norm = float(block_orig.norm().clamp_min(1e-8).item())
    pert_norm = float(block_diff.norm().item())
    # Per-channel z-score of perturbed block
    full_mean = ys_base[:, ch0:].float().mean(dim=(2, 3, 4), keepdim=True)
    full_std = ys_base[:, ch0:].float().std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
    z = ((block_pert - full_mean) / full_std).abs().mean()
    return {
        "perturbation_norm": pert_norm,
        "perturbation_norm_relative": pert_norm / base_norm,
        "block_zscore_mean": float(z.item()),
    }


# ============================================================================
# Decoded visual sanity
# ============================================================================


def _try_decode_ys_first_frame(head: Any, ys_full: torch.Tensor) -> np.ndarray | None:
    """Decode the first time slice of ys (latent channels only) into RGB.

    ys layout: channels [0..3] = mask, channels [4..] = VAE latent (z_dim).
    For decoding we strip the mask and expand T to whatever the VAE expects.
    Best-effort; returns None on failure.
    """
    try:
        z = ys_full[:, 4:, :, :, :].float()
        # The VAE.decode expects (B, C_lat, T_lat, H_lat, W_lat).
        with torch.inference_mode():
            params = next(head.vae.parameters(), None)
            if params is None:
                return None
            z = z.to(params.device, dtype=params.dtype)
            frames = head.vae.decode(
                z,
                tiled=getattr(head, "tiled", True),
                tile_size=(getattr(head, "tile_size_height", 34), getattr(head, "tile_size_width", 34)),
                tile_stride=(getattr(head, "tile_stride_height", 18), getattr(head, "tile_stride_width", 16)),
            )
        # frames: (B, C, T, H, W) in [-1, 1]
        f0 = frames[0, :, 0]                     # (C, H, W)
        f0 = ((f0.float() + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        return np.transpose(f0, (1, 2, 0))       # H, W, C
    except Exception as e:
        logger.debug("VAE decode failed: %s", e)
        return None


def _save_decoded_grid(head: Any, ys_base: torch.Tensor, ys_pert: torch.Tensor,
                       out_path: Path, title: str) -> bool:
    base_img = _try_decode_ys_first_frame(head, ys_base)
    pert_img = _try_decode_ys_first_frame(head, ys_pert)
    if base_img is None or pert_img is None:
        return False
    diff = np.abs(base_img.astype(np.int16) - pert_img.astype(np.int16)).clip(0, 255).astype(np.uint8)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
        for ax, img, lab in zip(axes, [base_img, pert_img, diff], ["baseline", "perturbed", "|diff|"]):
            ax.imshow(img); ax.set_title(lab, fontsize=9); ax.axis("off")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        logger.warning("Decoded grid save failed: %s", e)
        return False


# ============================================================================
# Aux. inputs computation: dataset mean and prev-frame ys
# ============================================================================


def compute_dataset_mean(baselines: list[BaselineCache]) -> torch.Tensor | None:
    """Per-channel mean of the latent (channels [ch0:]) across all baselines, broadcast-ready."""
    lat_means = []
    for b in baselines:
        ch0 = b.z_channel_start
        m = b.ys[:, ch0:].float().mean(dim=(2, 3, 4))   # (B, C)
        lat_means.append(m)
    if not lat_means:
        return None
    stack = torch.stack(lat_means, dim=0)  # (E, B, C)
    return stack.mean(dim=(0, 1)).reshape(1, -1, 1, 1, 1)


def fetch_prev_frame_ys(policy: GrootSimPolicy, manifest_entry: dict[str, Any],
                        lag: int, seed: int) -> torch.Tensor | None:
    """Build an obs at frame_index - lag from the same episode, run encode_image, return cached ys."""
    if manifest_entry.get("frame_index", 0) - lag < 0:
        return None
    fake_entry = dict(manifest_entry)
    fake_entry["frame_index"] = manifest_entry["frame_index"] - lag
    try:
        obs = build_obs(fake_entry)
    except Exception as e:
        logger.warning("Could not build prev-frame obs (lag=%d): %s", lag, e)
        return None
    head = policy.trained_model.action_head
    cache: dict[str, Any] = {}
    with _patched_encode_image(head, capturing=True, cache=cache):
        try:
            _, _, _, _ = _run_with_seed(policy, obs, seed)
        except Exception as e:
            logger.warning("Prev-frame forward failed (lag=%d): %s", lag, e)
            return None
    return cache.get("ys")


# ============================================================================
# Driver: per-example sweep
# ============================================================================


def sweep_example(policy: GrootSimPolicy,
                  baseline: BaselineCache,
                  manifest_entry: dict[str, Any],
                  regions: list[str],
                  dataset_mean_lat: torch.Tensor | None,
                  noise_floor_overall: float | None,
                  out_dir: Path,
                  seed_global: int,
                  decode_examples: int) -> list[dict[str, Any]]:
    head = policy.trained_model.action_head
    rng = np.random.default_rng(seed_global)
    rows: list[dict[str, Any]] = []

    # Lazy fetch of previous-frame ys
    prev_ys_cache: dict[str, torch.Tensor | None] = {}

    def _get_prev_ys(lag: int) -> torch.Tensor | None:
        key = f"prev_ys_l{lag}"
        if key not in prev_ys_cache:
            prev_ys_cache[key] = fetch_prev_frame_ys(policy, manifest_entry, lag, baseline.seed)
        return prev_ys_cache[key]

    decoded_count = 0
    for op in OPERATORS:
        # Inputs needed for this operator
        op_kwargs: dict[str, Any] = dict(op.extras)
        if "dataset_mean_lat" in op.needs:
            if dataset_mean_lat is None:
                logger.info("  skipping %s (dataset_mean_lat unavailable)", op.name)
                continue
            op_kwargs["dataset_mean_lat"] = dataset_mean_lat
        for need in op.needs:
            if need.startswith("prev_ys_l"):
                lag = int(need.split("l")[-1])
                prev = _get_prev_ys(lag)
                if prev is None:
                    logger.info("  skipping %s (no prev frame at lag %d)", op.name, lag)
                    op_kwargs = None
                    break
                op_kwargs["prev_ys"] = prev
        if op_kwargs is None:
            continue

        op_kwargs["rng"] = rng

        for strength in op.strengths:
            for region_name in regions:
                region = _region_slices(tuple(baseline.ys.shape), region_name, rng)

                t_start = time.perf_counter()
                try:
                    ys_pert = op.fn(baseline.ys.clone(), region, baseline.z_channel_start,
                                    strength=strength, **op_kwargs)
                except Exception as e:
                    logger.warning("  %s op_fn failed: %s", op.name, e)
                    continue
                construct_ms = (time.perf_counter() - t_start) * 1000.0

                # Run model with perturbed ys
                try:
                    pert_chunk, pert_actions, pert_video, elapsed = run_perturbed(policy, baseline, ys_pert)
                except Exception as e:
                    logger.warning("  %s perturbed fwd failed: %s", op.name, e)
                    continue

                # Metrics
                am = _action_metrics(baseline.action_chunk, pert_chunk)
                vp_l2 = _video_metrics(baseline.video_pred, pert_video)
                lm = _latent_metrics(baseline.ys, ys_pert, region, baseline.z_channel_start)

                has_nan = bool(np.any(np.isnan(pert_chunk))) if pert_chunk.size else False
                normalized_effect = (
                    am["delta_l2"] / (noise_floor_overall + 1e-8)
                    if (noise_floor_overall is not None and not math.isnan(am["delta_l2"]))
                    else float("nan")
                )

                row = {
                    "example_id": manifest_entry["example_id"],
                    "task_group": manifest_entry.get("task_group", ""),
                    "operator": op.name,
                    "category": op.category,
                    "strength": strength,
                    "region": region_name,
                    "elapsed_s": elapsed,
                    "construct_ms": construct_ms,
                    "has_nan": int(has_nan),
                    "valid": int(am["shape_match"] and not has_nan),
                    **am,
                    "delta_video_pred_l2": vp_l2,
                    **lm,
                    "normalized_effect": normalized_effect,
                }
                rows.append(row)

                # Decode visual sanity for the first N (op, strength, region) hits per example.
                if decoded_count < decode_examples and region_name in ("whole_grid", "center", "bottom_center"):
                    title = f"{manifest_entry['example_id']} | {op.name} α={strength} {region_name}"
                    out_path = out_dir / "decoded_grid" / manifest_entry["example_id"] / \
                        f"{op.name}__a{strength}__{region_name}.png"
                    if _save_decoded_grid(head, baseline.ys, ys_pert, out_path, title):
                        decoded_count += 1

                logger.info("  op=%s α=%g region=%s   Δa_l2=%.5f Δa_first=%.5f normΔz=%.3f valid=%d",
                            op.name, strength, region_name,
                            am["delta_l2"], am["delta_first"], lm["perturbation_norm_relative"],
                            row["valid"])
    return rows


# ============================================================================
# Aggregation, ranking, plots, report
# ============================================================================


def aggregate_per_operator(rows: list[dict[str, Any]],
                           noise_floor: float | None) -> dict[str, Any]:
    by_op: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_op.setdefault(r["operator"], []).append(r)

    summary: dict[str, Any] = {}
    for op_name, rs in by_op.items():
        if not rs:
            continue
        cat = rs[0]["category"]
        delta_l2 = [r["delta_l2"] for r in rs if not math.isnan(r["delta_l2"])]
        valid_rate = float(np.mean([r["valid"] for r in rs]))
        nan_rate = float(np.mean([r["has_nan"] for r in rs]))
        # Locality proxy: ratio of action delta when the region is small (center / corners)
        # vs. when whole_grid is perturbed. Lower = more localized.
        region_means: dict[str, list[float]] = {}
        for r in rs:
            region_means.setdefault(r["region"], []).append(r["delta_l2"])
        whole = float(np.mean(region_means.get("whole_grid", [0.0]))) if region_means.get("whole_grid") else float("nan")
        small_regions = [v for k, lst in region_means.items() if k != "whole_grid" for v in lst]
        small_mean = float(np.mean(small_regions)) if small_regions else float("nan")
        locality = (small_mean / whole) if (whole and not math.isnan(whole) and whole > 1e-8) else float("nan")

        # Sensitivity: median delta_l2
        sensitivity = float(np.median(delta_l2)) if delta_l2 else 0.0
        # In-distribution proxy: median latent z-score (lower = closer to natural distribution)
        z_scores = [r["block_zscore_mean"] for r in rs]
        ood_z = float(np.median(z_scores)) if z_scores else float("nan")
        # Decoded artifact severity is captured visually only; we also track perturbation_norm.
        norm_rel = [r["perturbation_norm_relative"] for r in rs]
        norm_rel_med = float(np.median(norm_rel)) if norm_rel else float("nan")

        # Significance vs noise floor
        if noise_floor and noise_floor > 0 and delta_l2:
            sig_frac = float(np.mean([d > 2.0 * noise_floor for d in delta_l2]))
        else:
            sig_frac = float("nan")

        summary[op_name] = {
            "category": cat,
            "n_runs": len(rs),
            "valid_rate": valid_rate,
            "nan_rate": nan_rate,
            "delta_l2_mean": float(np.mean(delta_l2)) if delta_l2 else 0.0,
            "delta_l2_median": sensitivity,
            "delta_l2_p90": float(np.quantile(delta_l2, 0.90)) if delta_l2 else 0.0,
            "locality_ratio_small_over_whole": locality,
            "ood_block_zscore_median": ood_z,
            "perturbation_norm_relative_median": norm_rel_med,
            "fraction_above_2sigma_seed": sig_frac,
        }
    return summary


def rank_operators(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Compose a Stage-1 ranking score: sensitivity × validity × in-distribution."""
    ops = list(summary.items())
    if not ops:
        return []

    sens_arr = np.array([s.get("delta_l2_median", 0.0) for _, s in ops])
    val_arr = np.array([s.get("valid_rate", 0.0) for _, s in ops])
    ood_arr = np.array([s.get("ood_block_zscore_median", 0.0) for _, s in ops])
    loc_arr = np.array([
        (1.0 - s.get("locality_ratio_small_over_whole", 1.0))
        if not math.isnan(s.get("locality_ratio_small_over_whole", float("nan"))) else 0.0
        for _, s in ops
    ])

    # Normalise to [0, 1]
    def _norm(a: np.ndarray, invert: bool = False) -> np.ndarray:
        if a.size == 0:
            return a
        m = float(np.nanmin(a)); M = float(np.nanmax(a))
        if M - m < 1e-12:
            return np.zeros_like(a)
        x = (a - m) / (M - m)
        return 1.0 - x if invert else x

    sens_n = _norm(sens_arr)
    val_n = _norm(val_arr)
    ood_n = _norm(ood_arr, invert=True)        # lower OOD = better
    loc_n = _norm(loc_arr)                     # higher locality = better

    rank_score = 0.4 * sens_n + 0.2 * val_n + 0.3 * ood_n + 0.1 * loc_n
    out = []
    for (name, s), score in zip(ops, rank_score):
        out.append({
            "operator": name,
            "category": s.get("category", ""),
            "rank_score": float(score),
            "sensitivity_norm": float(sens_n[ops.index((name, s))]),
            "validity_norm": float(val_n[ops.index((name, s))]),
            "ood_norm": float(ood_n[ops.index((name, s))]),
            "locality_norm": float(loc_n[ops.index((name, s))]),
            **{k: v for k, v in s.items() if k != "category"},
        })
    out.sort(key=lambda r: r["rank_score"], reverse=True)
    return out


# ----- Plots ----------------------------------------------------------------


def _save_fig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(str(base) + ext)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_strength_sweeps(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    by_op: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_op.setdefault(r["operator"], []).append(r)
    for op, rs in by_op.items():
        strengths = sorted({r["strength"] for r in rs})
        if len(strengths) < 2:
            continue
        regions = sorted({r["region"] for r in rs})
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        for i, region in enumerate(regions):
            xs, ys, ys_lo, ys_hi = [], [], [], []
            for s in strengths:
                vals = [r["delta_l2"] for r in rs if r["strength"] == s and r["region"] == region]
                if not vals:
                    continue
                xs.append(s)
                ys.append(float(np.mean(vals)))
                ys_lo.append(float(np.quantile(vals, 0.10)) if len(vals) >= 2 else float(np.mean(vals)))
                ys_hi.append(float(np.quantile(vals, 0.90)) if len(vals) >= 2 else float(np.mean(vals)))
            if xs:
                ax.plot(xs, ys, "o-", color=cmap(i % 10), label=region)
                ax.fill_between(xs, ys_lo, ys_hi, color=cmap(i % 10), alpha=0.15)
        ax.set_xlabel("strength")
        ax.set_ylabel(r"$\Delta a_{L_2}$")
        ax.set_title(f"E0.6 — strength sweep: {op}")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        _save_fig(fig, out_dir / "plots" / f"plot_strength_sweep__{op}")


def plot_operator_heatmap(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not rows:
        return
    ops = sorted({r["operator"] for r in rows})
    regions = sorted({r["region"] for r in rows})
    M = np.zeros((len(ops), len(regions)))
    for i, op in enumerate(ops):
        for j, region in enumerate(regions):
            vals = [r["delta_l2"] for r in rows if r["operator"] == op and r["region"] == region]
            M[i, j] = float(np.mean(vals)) if vals else float("nan")

    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(regions) + 2.0),
                                    max(3.0, 0.3 * len(ops) + 1.0)))
    im = ax.imshow(M, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=20)
    ax.set_yticks(range(len(ops))); ax.set_yticklabels(ops, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"mean $\Delta a_{L_2}$")
    ax.set_title("E0.6 — operator × region action delta")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plots" / "plot_operator_heatmap")


def plot_validity(summary: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not summary:
        return
    ops = list(summary.keys())
    valid = [summary[o]["valid_rate"] for o in ops]
    nan_rate = [summary[o]["nan_rate"] for o in ops]
    x = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(ops) + 3.0), 3.2))
    ax.bar(x - 0.2, valid, width=0.4, color="#3b8b58", edgecolor="black", label="valid_rate")
    ax.bar(x + 0.2, nan_rate, width=0.4, color="#c14b4b", edgecolor="black", label="nan_rate")
    ax.set_xticks(x); ax.set_xticklabels(ops, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("E0.6 — operator validity / NaN rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_dir / "plots" / "plot_validity")


def plot_operator_ranking(ranking: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not ranking:
        return
    names = [r["operator"] for r in ranking]
    sens = [r["sensitivity_norm"] for r in ranking]
    val = [r["validity_norm"] for r in ranking]
    ood = [r["ood_norm"] for r in ranking]
    loc = [r["locality_norm"] for r in ranking]
    score = [r["rank_score"] for r in ranking]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(names) + 3.0), 3.6))
    width = 0.18
    ax.bar(x - 1.5 * width, sens, width, label="sensitivity", color="#3b75af")
    ax.bar(x - 0.5 * width, val, width, label="validity", color="#3b8b58")
    ax.bar(x + 0.5 * width, ood, width, label="in-distribution (1−OOD)", color="#7a4dab")
    ax.bar(x + 1.5 * width, loc, width, label="locality", color="#c4831f")
    ax.plot(x, score, "k--", lw=1.0, label="rank score")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("E0.6 — operator ranking (normalized criteria)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir / "plots" / "plot_operator_ranking")


# ----- Report ---------------------------------------------------------------


def write_combined_report(out_dir: Path, ranking: list[dict[str, Any]],
                          summary: dict[str, Any], num_examples: int,
                          regions: list[str]) -> None:
    lines = ["# Stage 0 / E0.6 — Perturbation Operator Suite",
             "",
             f"- Examples: {num_examples}",
             f"- Regions tested: {', '.join(regions)}",
             f"- Operators tested: {len(summary)}",
             "",
             "## Operator ranking (paper-ready)",
             "",
             "| Rank | Operator | Category | Score | Sensitivity (median Δa_L2) | "
             "Validity | In-distribution (1−OOD norm) | Locality | "
             "% Δa > 2 σ_seed |",
             "|----:|----------|----------|------:|---------------------------:|"
             "---------:|-----------------------------:|---------:|----------------:|"]
    for i, r in enumerate(ranking):
        sig = r.get("fraction_above_2sigma_seed", float("nan"))
        sig_str = f"{sig * 100:.0f}%" if (isinstance(sig, float) and not math.isnan(sig)) else "—"
        lines.append(
            f"| {i+1} | `{r['operator']}` | {r['category']} | {r['rank_score']:.3f} | "
            f"{r.get('delta_l2_median', float('nan')):.4f} | "
            f"{r.get('valid_rate', float('nan')):.2f} | "
            f"{r.get('ood_norm', float('nan')):.2f} | "
            f"{r.get('locality_norm', float('nan')):.2f} | "
            f"{sig_str} |"
        )
    lines += [
        "",
        "## Per-category figures",
        "",
        "- `plots/plot_operator_ranking.png` — composite ranking",
        "- `plots/plot_validity.png` — validity & NaN rates",
        "- `plots/plot_operator_heatmap.png` — operator × region action-delta heatmap",
        "- `plots/plot_strength_sweep__<operator>.png` — strength sweep per operator",
        "- `decoded_grid/<example>/<op>__α…__<region>.png` — visual sanity grid",
        "",
        "## Stage-1 recommendation",
        "",
        "Pick **2–3 operators that score well on validity AND in-distribution AND sensitivity**. "
        "Rules of thumb that follow from this suite:",
        "",
        "- Use `local_mean` (in-distribution) as the **primary causal operator** — it removes "
        "  region-specific information without injecting OOD content. ",
        "- Use `avg_pool_2` / `blur_k3` (compression) as the **compression-relevant operator** — "
        "  this is what CALA-WAM actually does in deployment. ",
        "- Use `prev_frame_cache_l1` (and `_l2`, `_l4` for cache-staleness sweeps) as the "
        "  **temporal operator** — directly tests static-vs-dynamic redundancy.",
        "- Keep `zero_mask` only as an **appendix stress test**: large action deltas, but "
        "  also large `block_zscore_mean` (OOD) and worse locality.",
        "",
        "If two operators disagree on which region matters, trust the in-distribution one. "
        "A perturbation is **significant** only when "
        "`delta_l2 > 2 × σ_seed` (column above), so always pair this suite with the "
        "`noise_floor.json` from Stage-0 E0.3.",
        "",
        "## Reproducibility",
        "",
        "- Same seed used for baseline and perturbed runs (paired-seed protocol).",
        "- VAE-encoded latents `head.ys` perturbed only in channels 4: (mask channels preserved).",
        "- Region semantics:",
    ]
    for r, m in REGION_MEANING.items():
        lines.append(f"  - `{r}` — {m}")
    (out_dir / "combined_report.md").write_text("\n".join(lines))


# ============================================================================
# Helpers / driver
# ============================================================================


def _write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        out.write_text("")
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _select_examples(manifest: list[dict[str, Any]], num: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    chosen: list[dict[str, Any]] = []
    for entry in manifest:
        g = entry.get("task_group", "")
        if g not in seen:
            chosen.append(entry); seen.add(g)
        if len(chosen) >= num:
            break
    if len(chosen) < num:
        for entry in manifest:
            if entry not in chosen:
                chosen.append(entry)
            if len(chosen) >= num:
                break
    return chosen[:num]


def _load_noise_floor(path: str | None) -> float | None:
    if not path:
        return None
    try:
        with Path(path).open("r") as f:
            data = json.load(f)
        v = data.get("sigma_seed_overall_mean") or data.get("sigma_seed_overall_p90")
        return float(v) if v is not None else None
    except Exception as e:
        logger.warning("Could not read noise_floor=%s: %s", path, e)
        return None


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_examples", type=int, default=4,
                   help="Examples to evaluate (spread across task groups).")
    p.add_argument("--seed", type=int, default=0,
                   help="Paired seed used for both baseline and perturbed runs.")
    p.add_argument("--noise_floor", default=None,
                   help="Path to noise_floor.json from E0.3. Used for normalized_effect column.")
    p.add_argument("--regions", default="whole_grid,center,top_left,bottom_right,bottom_center,random_block",
                   help="Comma-separated region presets.")
    p.add_argument("--decoded_per_example", type=int, default=6,
                   help="Number of decoded sanity grids to save per example.")
    p.add_argument("--smoke", action="store_true",
                   help="Use shorter strength sweeps and only 2 regions for quick iteration.")
    args = p.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_neurips_matplotlib()

    if args.smoke:
        for op in OPERATORS:
            op.strengths = [op.strengths[-1]]
        regions = [r for r in args.regions.split(",") if r.strip()][:2] or ["whole_grid", "center"]
    else:
        regions = [r for r in args.regions.split(",") if r.strip()]

    with Path(args.task_suite).resolve().open("r") as f:
        manifest = json.load(f)
    chosen = _select_examples(manifest, args.num_examples)
    if not chosen:
        logger.error("Manifest is empty.")
        sys.exit(1)

    init_dist_single_process()
    logger.info("Loading model from %s ...", args.checkpoint)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        model_path=args.checkpoint,
        device=args.device,
    )
    logger.info("Model loaded.")

    noise_floor = _load_noise_floor(args.noise_floor)
    if noise_floor is not None:
        logger.info("Using σ_seed=%.6f for normalized_effect", noise_floor)
    else:
        logger.info("No noise_floor provided — `normalized_effect` will be nan.")
    logger.info("Using regions: %s", regions)                                               
    # 1. Baseline pass for every example (gives us ys, video_pred, action_chunk).
    baselines: list[BaselineCache] = []
    entries: list[dict[str, Any]] = []
    for entry in chosen:
        try:
            obs = build_obs(entry)
        except Exception as e:
            logger.warning("Skipping %s: %s", entry.get("example_id"), e)
            continue
        logger.info("Capturing baseline for %s ...", entry["example_id"])
        try:
            cache = capture_baseline(policy, obs, args.seed)
        except Exception as e:
            logger.warning("Baseline capture failed for %s: %s", entry["example_id"], e)
            continue
        baselines.append(cache)
        entries.append(entry)

    if not baselines:
        logger.error("No usable baselines.")
        sys.exit(1)

    # 2. Compute auxiliary inputs (dataset_mean_lat).
    dataset_mean_lat = compute_dataset_mean(baselines)
    if dataset_mean_lat is not None:
        logger.info("Computed dataset_mean_lat shape=%s", list(dataset_mean_lat.shape))

    # 3. Sweep operators × strengths × regions for every example.
    all_rows: list[dict[str, Any]] = []
    for baseline, entry in zip(baselines, entries):
        logger.info("Sweeping example %s ...", entry["example_id"])
        rows = sweep_example(
            policy, baseline, entry, regions,
            dataset_mean_lat=dataset_mean_lat,
            noise_floor_overall=noise_floor,
            out_dir=out_dir,
            seed_global=args.seed,
            decode_examples=args.decoded_per_example,
        )
        all_rows.extend(rows)

    _write_csv(all_rows, out_dir / "per_run.csv")

    summary = aggregate_per_operator(all_rows, noise_floor)
    (out_dir / "summary_per_operator.json").write_text(json.dumps(summary, indent=2))

    ranking = rank_operators(summary)
    _write_csv(ranking, out_dir / "operator_ranking.csv")

    # 4. Plots
    plot_strength_sweeps(all_rows, out_dir)
    plot_operator_heatmap(all_rows, out_dir)
    plot_validity(summary, out_dir)
    plot_operator_ranking(ranking, out_dir)

    write_combined_report(out_dir, ranking, summary, len(baselines), regions)
    (out_dir / "combined_report.json").write_text(json.dumps({
        "examples": [b.obs.get("annotation.language.action_text", "") if isinstance(b.obs, dict) else ""
                     for b in baselines],
        "regions": regions,
        "noise_floor_used": noise_floor,
        "summary": summary,
        "ranking": ranking,
    }, indent=2))

    logger.info("Done. Combined report -> %s", out_dir / "combined_report.md")


if __name__ == "__main__":
    main()
