#!/usr/bin/env python3
"""Stage 3 / Compute — CALA-WAM matched-budget allocation sweep (E3.1, E3.2).

Implements the **method evaluation pass**. For every example × method × budget
× variant we (a) build a retention mask using that method's importance map,
(b) compress the *non-retained* latent blocks with a chosen operator, (c) run
one perturbed forward, and (d) record action / video error vs the unperturbed
baseline + an approximate latency proxy.

Methods (each produces a 2-D importance map over the latent grid):

    random              — i.i.d. uniform per cell.
    uniform             — flat (every cell equal; budget chooses a regular grid).
    center              — analytic Gaussian, image-centre.
    gripper             — analytic Gaussian, bottom-centre proxy.
    object              — alias for `clip_features` from Stage 2.
    flow                — alias for `optical_flow` from Stage 2.
    attention           — alias for `dit_attention` from Stage 2.
    edges               — alias for `edges_input` from Stage 2.
    action_causal       — Stage-1 primary action heatmap (CALA-WAM signal).
    cala_wam_hybrid     — α · action_causal + (1-α) · video_causal,
                          combining action and future-video sensitivity.

Variants (how to compress the *non-retained* blocks):

    A_hard_retention    — full identity for top-k, local_mean for rest.
    B_soft_fidelity     — three tiers: identity > blur_k3 > local_mean.
    C_temporal_cache    — non-retained blocks replaced by previous-frame ys.
    D_global_summary    — non-retained blocks replaced by their pooled mean.

Each row in the output table records:

    example_id, task_group, role, method, variant, budget_pct,
    fraction_retained_actual, action_l2, action_first, action_gripper,
    action_cosine, video_l2, perturbation_norm_relative,
    forward_elapsed_s, approx_latency_ratio = budget_pct/100,
    valid (no NaN, shapes match)

Example:

    python scripts/stage3/allocation_compute.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --maps_dir runs/stage1_maps \\
        --saliency_dir runs/stage2_saliency \\
        --num_examples 12 \\
        --budgets 100,75,50,25,12.5 \\
        --methods action_causal,cala_wam_hybrid,random,uniform,center,gripper,object,flow,attention \\
        --variants A_hard_retention,B_soft_fidelity,C_temporal_cache \\
        --output_dir runs/stage3_alloc

The budget=100 column = baseline action error = 0; we keep it as a calibration row.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage3.compute")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

import torch.nn.functional as F  # noqa: E402
from tianshou.data import Batch  # noqa: E402

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
sys.path.insert(0, str(STAGE0_DIR))
from _common import (  # noqa: E402
    EMBODIMENT_TAG, build_obs, init_dist_single_process,
)

import perturbation_suite as ps  # noqa: E402


# ============================================================================
# Importance-map providers per method
# ============================================================================


def _gaussian_2d(shape: tuple[int, int], cy_frac: float, cx_frac: float,
                 sigma_frac: float) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cy = cy_frac * (H - 1); cx = cx_frac * (W - 1)
    sy = max(1.0, sigma_frac * H); sx = max(1.0, sigma_frac * W)
    return np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2))).astype(np.float32)


def _broadcast_3d(arr2: np.ndarray, T: int) -> np.ndarray:
    return np.broadcast_to(arr2[None, :, :], (T, *arr2.shape)).copy()


def _load_method_map(method: str,
                     latent_shape: tuple[int, int, int],
                     stage1_arrays: dict[str, np.ndarray] | None,
                     saliency_arrays: dict[str, np.ndarray] | None,
                     hybrid_alpha: float,
                     primary_op: str,
                     primary_metric: str,
                     primary_video_metric: str,
                     rng: np.random.Generator) -> np.ndarray | None:
    """Return a (T, H, W) importance map for `method` (higher = more important)."""
    T, H, W = latent_shape

    if method == "random":
        return rng.uniform(0, 1, size=(T, H, W)).astype(np.float32)

    if method == "uniform":
        return np.ones((T, H, W), dtype=np.float32)

    if method == "center":
        return _broadcast_3d(_gaussian_2d((H, W), 0.5, 0.5, 0.18), T)

    if method == "gripper":
        return _broadcast_3d(_gaussian_2d((H, W), 0.78, 0.5, 0.16), T)

    if method in ("object", "flow", "attention", "edges"):
        if not saliency_arrays:
            return None
        key = {"object": "clip_features", "flow": "optical_flow",
               "attention": "dit_attention", "edges": "edges_input"}[method]
        sal = saliency_arrays.get(key)
        if sal is None:
            return None
        return sal.astype(np.float32)

    if method == "action_causal":
        if not stage1_arrays:
            return None
        return stage1_arrays.get(f"{primary_op}__{primary_metric}", np.empty(0)).astype(np.float32) \
            if f"{primary_op}__{primary_metric}" in stage1_arrays else None

    if method == "cala_wam_hybrid":
        if not stage1_arrays:
            return None
        a = stage1_arrays.get(f"{primary_op}__{primary_metric}")
        v = stage1_arrays.get(f"{primary_op}__{primary_video_metric}")
        if a is None and v is None:
            return None

        def _norm(x: np.ndarray | None) -> np.ndarray | None:
            if x is None:
                return None
            x = np.nan_to_num(x.astype(np.float32), nan=0.0)
            mn = float(x.min()); mx = float(x.max())
            if mx - mn < 1e-12:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        a_n = _norm(a); v_n = _norm(v)
        if a_n is None:
            return v_n
        if v_n is None:
            return a_n
        return (hybrid_alpha * a_n + (1.0 - hybrid_alpha) * v_n).astype(np.float32)

    return None


# ============================================================================
# Retention mask + compression variants
# ============================================================================


def _retention_mask_top_k(importance: np.ndarray, budget_pct: float) -> np.ndarray:
    """Boolean mask: True where retained (top-k%)."""
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


def _local_outside_mean(lat_ch: torch.Tensor, retain_mask_3d: torch.Tensor) -> torch.Tensor:
    """Per-channel mean of latents *inside* retained region (used as fallback fill).

    Used only as a fill value for non-retained cells when the full latent of a
    given example has no other natural reference.
    """
    full = lat_ch.float()
    keep = retain_mask_3d.float()                            # (T, H, W)
    keep_5d = keep[None, None, :, :, :].to(full.dtype)
    denom = keep.sum().clamp_min(1.0)
    mean_inside = (full * keep_5d).sum(dim=(2, 3, 4), keepdim=True) / denom
    return mean_inside


def _apply_variant_A(ys_base: torch.Tensor, retain_mask: np.ndarray,
                     ch0: int, **_kw) -> torch.Tensor:
    """Hard retention: identity inside retained mask, local_mean elsewhere."""
    mask_ch, lat_ch = _split_channels(ys_base, ch0)
    keep3 = torch.from_numpy(retain_mask.astype(np.float32)).to(lat_ch.device)
    keep5 = keep3[None, None, :, :, :].to(lat_ch.dtype)
    fill = _local_outside_mean(lat_ch, torch.from_numpy(retain_mask).to(lat_ch.device))
    new_lat = lat_ch * keep5 + fill * (1.0 - keep5)
    return _join_channels(mask_ch.clone(), new_lat)


def _apply_variant_B(ys_base: torch.Tensor, importance: np.ndarray,
                     budget_pct: float, ch0: int, **_kw) -> torch.Tensor:
    """Soft fidelity: top 1/3 of budget → identity; next 1/3 → blur_k3;
    bottom 1/3 of budget → local_mean. Non-retained → local_mean."""
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
    # Blur layer (depthwise 3×3)
    B, C, T, H, W = full.shape
    weight = torch.ones((C, 1, 1, 3, 3), dtype=full.dtype, device=full.device) / 9.0
    blurred = F.conv3d(full, weight, padding=(0, 1, 1), groups=C)
    fill = _local_outside_mean(full, torch.from_numpy(keep_mask).to(full.device))

    keep_t = torch.from_numpy(high_mask.astype(np.float32)).to(full.device)[None, None]
    mid_t = torch.from_numpy(mid_mask.astype(np.float32)).to(full.device)[None, None]
    low_t = torch.from_numpy(low_mask.astype(np.float32)).to(full.device)[None, None]
    drop_t = torch.from_numpy(drop_mask.astype(np.float32)).to(full.device)[None, None]
    new_lat = full * keep_t + blurred * mid_t + fill * (low_t + drop_t)
    return _join_channels(mask_ch.clone(), new_lat.to(lat_ch.dtype))


def _apply_variant_C(ys_base: torch.Tensor, retain_mask: np.ndarray,
                     ch0: int, prev_ys: torch.Tensor | None = None, **_kw) -> torch.Tensor:
    """Temporal cache: identity inside retained, prev_frame_ys outside."""
    if prev_ys is None:
        return _apply_variant_A(ys_base, retain_mask, ch0)
    mask_ch, lat_ch = _split_channels(ys_base, ch0)
    _, prev_lat = _split_channels(prev_ys.to(ys_base.device), ch0)
    keep3 = torch.from_numpy(retain_mask.astype(np.float32)).to(lat_ch.device)
    keep5 = keep3[None, None, :, :, :].to(lat_ch.dtype)
    new_lat = lat_ch * keep5 + prev_lat.to(lat_ch.dtype) * (1.0 - keep5)
    return _join_channels(mask_ch.clone(), new_lat)


def _apply_variant_D(ys_base: torch.Tensor, retain_mask: np.ndarray,
                     ch0: int, **_kw) -> torch.Tensor:
    """Global summary: replace all non-retained blocks with one global mean.

    Equivalent to variant A under the current single-summary reduction; kept
    as a separate label so analyses can distinguish them if local pooling is
    introduced later.
    """
    return _apply_variant_A(ys_base, retain_mask, ch0)


def _build_perturbed_ys(variant: str, ys_base: torch.Tensor, importance: np.ndarray,
                        budget_pct: float, ch0: int,
                        prev_ys: torch.Tensor | None) -> torch.Tensor:
    if budget_pct >= 100.0 - 1e-6:
        return ys_base.clone()
    retain = _retention_mask_top_k(importance, budget_pct)
    if variant == "A_hard_retention":
        return _apply_variant_A(ys_base, retain, ch0)
    if variant == "B_soft_fidelity":
        return _apply_variant_B(ys_base, importance, budget_pct, ch0)
    if variant == "C_temporal_cache":
        return _apply_variant_C(ys_base, retain, ch0, prev_ys=prev_ys)
    if variant == "D_global_summary":
        return _apply_variant_D(ys_base, retain, ch0)
    raise ValueError(f"Unknown variant {variant}")


# ============================================================================
# Metrics
# ============================================================================


def _action_metrics(base_chunk: np.ndarray, pert_chunk: np.ndarray) -> dict[str, float]:
    if base_chunk.shape != pert_chunk.shape or base_chunk.size == 0:
        return {"action_l2": float("nan"), "action_first": float("nan"),
                "action_cosine": float("nan"), "action_gripper": float("nan"),
                "action_near": float("nan"), "action_far": float("nan")}
    H, A = base_chunk.shape
    third = max(1, H // 3)
    diff = base_chunk - pert_chunk
    l2 = float(np.sqrt((diff ** 2).mean()))
    first = float(np.sqrt((diff[:1] ** 2).mean()))
    near = float(np.sqrt((diff[:third] ** 2).mean()))
    far = float(np.sqrt((diff[-third:] ** 2).mean()))
    grip = float(np.sqrt((diff[:, -1:] ** 2).mean()))
    a = base_chunk.flatten().astype(np.float64)
    b = pert_chunk.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    cos = float("nan") if na == 0 or nb == 0 else float(1.0 - np.dot(a, b) / (na * nb))
    return {"action_l2": l2, "action_first": first, "action_near": near,
            "action_far": far, "action_gripper": grip, "action_cosine": cos}


def _video_l2(vp_base: torch.Tensor | None, vp_pert: torch.Tensor | None) -> float:
    if vp_base is None or vp_pert is None or vp_base.shape != vp_pert.shape:
        return float("nan")
    return float(torch.sqrt(((vp_base.float() - vp_pert.float()) ** 2).mean()).item())


def _latent_perturbation_norm(ys_base: torch.Tensor, ys_pert: torch.Tensor, ch0: int) -> float:
    a = ys_base[:, ch0:].float()
    b = ys_pert[:, ch0:].float()
    base_norm = float(a.norm().clamp_min(1e-8).item())
    diff_norm = float((b - a).norm().item())
    return diff_norm / base_norm


# ============================================================================
# Per-example sweep
# ============================================================================


def sweep_example(policy: GrootSimPolicy,
                  entry: dict[str, Any],
                  stage1_arrays: dict[str, np.ndarray] | None,
                  saliency_arrays: dict[str, np.ndarray] | None,
                  methods: list[str], budgets: list[float], variants: list[str],
                  primary_op: str, primary_metric: str,
                  primary_video_metric: str,
                  hybrid_alpha: float, seed: int,
                  out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head

    # Baseline (also gives us ys + action chunk + video_pred)
    obs = build_obs(entry)
    baseline = ps.capture_baseline(policy, obs, seed)
    T_lat, H_lat, W_lat = int(baseline.ys.shape[2]), int(baseline.ys.shape[3]), int(baseline.ys.shape[4])
    latent_shape = (T_lat, H_lat, W_lat)
    rng = np.random.default_rng(seed)

    # Compute importance maps once per method
    importance_per_method: dict[str, np.ndarray] = {}
    for m in methods:
        imp = _load_method_map(
            m, latent_shape, stage1_arrays, saliency_arrays, hybrid_alpha,
            primary_op, primary_metric, primary_video_metric, rng,
        )
        if imp is None:
            logger.info("[%s] method %s unavailable (missing source map)", entry["example_id"], m)
            continue
        if imp.shape != latent_shape:
            try:
                if imp.ndim == 3 and imp.shape[1:] == latent_shape[1:]:
                    pass
                else:
                    # Resize 2D map to (H_lat, W_lat) and broadcast over T_lat
                    arr2 = np.nanmean(imp, axis=0) if imp.ndim == 3 else imp
                    import cv2
                    arr2 = cv2.resize(arr2.astype(np.float32),
                                      (W_lat, H_lat), interpolation=cv2.INTER_LINEAR)
                    imp = _broadcast_3d(arr2, T_lat)
            except Exception as e:
                logger.warning("[%s] method %s shape mismatch (%s vs %s): %s",
                               entry["example_id"], m, imp.shape, latent_shape, e)
                continue
        importance_per_method[m] = imp.astype(np.float32)

    # Optional: prev-frame ys for variant C
    prev_ys: torch.Tensor | None = None
    if "C_temporal_cache" in variants:
        prev_ys = ps.fetch_prev_frame_ys(policy, entry, lag=1, seed=seed)

    rows: list[dict[str, Any]] = []
    for method, imp in importance_per_method.items():
        for budget in budgets:
            for variant in variants:
                if variant == "C_temporal_cache" and prev_ys is None and budget < 100.0 - 1e-6:
                    continue
                t0 = time.perf_counter()
                try:
                    ys_pert = _build_perturbed_ys(variant, baseline.ys.clone(),
                                                  imp, budget, baseline.z_channel_start,
                                                  prev_ys=prev_ys)
                except Exception as e:
                    logger.warning("[%s] %s/%s/%g build failed: %s",
                                   entry["example_id"], method, variant, budget, e)
                    continue
                construct_s = time.perf_counter() - t0

                try:
                    pert_chunk, _pert_actions, pert_video, fwd_s = ps.run_perturbed(
                        policy, baseline, ys_pert
                    )
                except Exception as e:
                    logger.warning("[%s] %s/%s/%g forward failed: %s",
                                   entry["example_id"], method, variant, budget, e)
                    continue

                am = _action_metrics(baseline.action_chunk, pert_chunk)
                v_l2 = _video_l2(baseline.video_pred, pert_video)
                pn = _latent_perturbation_norm(baseline.ys, ys_pert, baseline.z_channel_start)

                retain_frac = float(_retention_mask_top_k(imp, budget).mean())
                rows.append({
                    "example_id": entry["example_id"],
                    "task_group": entry.get("task_group"),
                    "role": entry.get("role"),
                    "method": method,
                    "variant": variant,
                    "budget_pct": float(budget),
                    "fraction_retained_actual": retain_frac,
                    "approx_latency_ratio": float(budget) / 100.0,
                    "construct_s": float(construct_s),
                    "forward_elapsed_s": float(fwd_s),
                    **am,
                    "video_l2": v_l2,
                    "perturbation_norm_relative": pn,
                    "valid": int(not (
                        math.isnan(am["action_l2"]) or
                        am["action_l2"] is None or
                        np.any(np.isnan(pert_chunk))
                    )),
                })

    # Persist per-example rows
    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with (out_dir / "rows.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    (out_dir / "meta.json").write_text(json.dumps({
        "example_id": entry["example_id"],
        "task_group": entry.get("task_group"),
        "role": entry.get("role"),
        "instruction": entry.get("instruction"),
        "latent_shape": [T_lat, H_lat, W_lat],
        "methods_used": list(importance_per_method.keys()),
        "budgets": budgets,
        "variants": variants,
        "n_rows": len(rows),
    }, indent=2))
    return rows


# ============================================================================
# Driver
# ============================================================================


def _select_examples(manifest: list[dict[str, Any]], num: int,
                     phase_balanced: bool) -> list[dict[str, Any]]:
    if not phase_balanced:
        return manifest[:num]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in manifest:
        by_key.setdefault((entry.get("task_group", ""), entry.get("role", "")), []).append(entry)
    keys = sorted(by_key.keys())
    chosen: list[dict[str, Any]] = []
    while len(chosen) < num and keys:
        for k in list(keys):
            if not by_key[k]:
                keys.remove(k); continue
            chosen.append(by_key[k].pop(0))
            if len(chosen) >= num:
                break
    return chosen[:num]


def _load_stage1_arrays(maps_dir: Path | None, example_id: str) -> dict[str, np.ndarray] | None:
    if maps_dir is None:
        return None
    npz = maps_dir / example_id / "heatmaps.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    return {k: z[k] for k in z.files}


def _load_saliency_arrays(saliency_dir: Path | None, example_id: str) -> dict[str, np.ndarray] | None:
    if saliency_dir is None:
        return None
    npz = saliency_dir / example_id / "saliency_maps.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    return {k: z[k] for k in z.files}


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True)
    p.add_argument("--maps_dir", required=True,
                   help="Stage-1 directory (causal_maps_compute output).")
    p.add_argument("--saliency_dir", default=None,
                   help="Stage-2 saliency directory (saliency_compute output).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_examples", type=int, default=12)
    p.add_argument("--phase_balanced", action="store_true")
    p.add_argument("--budgets", default="100,75,50,25,12.5",
                   help="Retention budgets in percent.")
    p.add_argument("--methods",
                   default="action_causal,cala_wam_hybrid,random,uniform,center,gripper,object,flow,attention",
                   help="Methods to evaluate.")
    p.add_argument("--variants",
                   default="A_hard_retention,B_soft_fidelity,C_temporal_cache",
                   help="Compression variants.")
    p.add_argument("--hybrid_alpha", type=float, default=0.7,
                   help="Weight on action_causal in cala_wam_hybrid (rest goes to video_causal).")
    p.add_argument("--primary_operator", default="local_mean")
    p.add_argument("--primary_metric", default="action_l2")
    p.add_argument("--primary_video_metric", default="video_l2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    with Path(args.task_suite).resolve().open("r") as f:
        manifest = json.load(f)
    chosen = _select_examples(manifest, args.num_examples, args.phase_balanced)
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

    maps_dir = Path(args.maps_dir).resolve() if args.maps_dir else None
    sal_dir = Path(args.saliency_dir).resolve() if args.saliency_dir else None

    all_rows: list[dict[str, Any]] = []
    for i, entry in enumerate(chosen):
        ex_dir = out_dir / entry["example_id"]
        if args.skip_existing and (ex_dir / "rows.csv").exists():
            logger.info("[%d/%d] %s skipped (already done)", i + 1, len(chosen), entry["example_id"])
            try:
                with (ex_dir / "rows.csv").open("r") as f:
                    rd = csv.DictReader(f)
                    for row in rd:
                        all_rows.append({k: _coerce(v) for k, v in row.items()})
            except Exception:
                pass
            continue
        stage1 = _load_stage1_arrays(maps_dir, entry["example_id"])
        sal = _load_saliency_arrays(sal_dir, entry["example_id"])
        if stage1 is None and any(m in ("action_causal", "cala_wam_hybrid") for m in methods):
            logger.warning("[%s] Stage-1 maps missing — action_causal / cala_wam_hybrid will be skipped",
                           entry["example_id"])
        if sal is None and any(m in ("object", "flow", "attention", "edges") for m in methods):
            logger.warning("[%s] Stage-2 saliency missing — proxy methods will be skipped",
                           entry["example_id"])

        logger.info("[%d/%d] sweeping %s methods=%d budgets=%d variants=%d",
                    i + 1, len(chosen), entry["example_id"],
                    len(methods), len(budgets), len(variants))
        try:
            rows = sweep_example(
                policy=policy, entry=entry,
                stage1_arrays=stage1, saliency_arrays=sal,
                methods=methods, budgets=budgets, variants=variants,
                primary_op=args.primary_operator, primary_metric=args.primary_metric,
                primary_video_metric=args.primary_video_metric,
                hybrid_alpha=args.hybrid_alpha, seed=args.seed,
                out_dir=ex_dir,
            )
            all_rows.extend(rows)
        except Exception as e:
            logger.error("[%s] sweep failed: %s", entry["example_id"], e, exc_info=True)
            continue

    # Aggregate
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    if fieldnames:
        with (out_dir / "all_rows.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
    (out_dir / "compute_run.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "maps_dir": str(maps_dir) if maps_dir else None,
        "saliency_dir": str(sal_dir) if sal_dir else None,
        "budgets": budgets,
        "methods": methods,
        "variants": variants,
        "hybrid_alpha": args.hybrid_alpha,
        "n_examples": len({r["example_id"] for r in all_rows}),
        "n_rows": len(all_rows),
    }, indent=2))
    logger.info("Done. all_rows.csv -> %s", out_dir / "all_rows.csv")


def _coerce(v: str) -> Any:
    try:
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return v


if __name__ == "__main__":
    main()
