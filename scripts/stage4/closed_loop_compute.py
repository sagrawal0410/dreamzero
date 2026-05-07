#!/usr/bin/env python3
"""Stage 4 / Compute — Closed-loop trajectory replay for CALA-WAM (E4.1 – E4.4).

Implements the **honest closed-loop proxy** for Stage 4. For every episode in
the manifest we sweep every (timestep × method × budget × variant) combination
and measure how much CALA-WAM (and every baseline) deviates from the
ground-truth action chunk at that step.

Why offline replay (vs full sim rollout)?

  * The ground truth is the recorded DROID demonstration action chunk.
  * Real closed-loop sim eval requires the `sim_evals` package + Isaac Lab,
    which is gated and slow. `scripts/stage4/cala_wam_policy.py` ships a
    minimal websocket-server adapter for users that have it; this script
    provides the deterministic, reproducible signal for the paper tables.

For each row we record:

    episode_index, frame_index, role, task_group, instruction,
    method, variant, budget_pct, fraction_retained_actual,
    forward_elapsed_s, approx_latency_ratio,
    action_l2, action_first, action_near, action_far,
    action_gripper, action_cosine,
    action_l2_vs_gt              # action error vs the demonstration GT
    gripper_event                # 1 if a gripper-state transition happens at this step
    valid

Trajectories are reconstructed by grouping manifest rows by `episode_index`
and ordering within an episode by `frame_index`. Stage-1 maps are looked up
per (episode_index, frame_index) using the example_id; if a map is missing
for a step we fall back to a same-episode neighbour or to a cheap proxy.

Example:

    python scripts/stage4/closed_loop_compute.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --maps_dir runs/stage1_maps \\
        --saliency_dir runs/stage2_saliency \\
        --num_episodes 6 \\
        --budgets 100,50,25 \\
        --methods action_causal,cala_wam_hybrid,uniform,attention,gripper \\
        --variants A_hard_retention \\
        --output_dir runs/stage4_closed_loop

Smoke (~10 min):

    python scripts/stage4/closed_loop_compute.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --maps_dir runs/stage1_maps \\
        --num_episodes 2 --budgets 50 --methods action_causal,uniform \\
        --output_dir runs/stage4_smoke
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage4.compute")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
STAGE3_DIR = SCRIPT_DIR.parent / "stage3"
sys.path.insert(0, str(STAGE0_DIR))
sys.path.insert(0, str(STAGE3_DIR))

from _common import (  # noqa: E402
    EMBODIMENT_TAG, build_obs, init_dist_single_process,
)
import perturbation_suite as ps  # noqa: E402
from allocation_compute import (  # noqa: E402
    _build_perturbed_ys,
    _load_method_map,
    _coerce,
)


# ============================================================================
# Manifest -> trajectories
# ============================================================================


def group_into_trajectories(manifest: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Return {episode_index: [entry, ...]} sorted by frame_index."""
    by_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest:
        ep = entry.get("episode_index")
        if ep is None:
            continue
        by_ep[int(ep)].append(entry)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda r: int(r.get("frame_index", 0)))
    return by_ep


def _select_episodes(by_ep: dict[int, list[dict[str, Any]]],
                     num: int, group_balanced: bool) -> list[int]:
    """Pick episode_indices, optionally stratified across task_groups."""
    if not group_balanced:
        return list(by_ep.keys())[:num]
    by_group: dict[str, list[int]] = defaultdict(list)
    for ep, entries in by_ep.items():
        # The episode's task group equals its first entry's task_group.
        g = (entries[0] or {}).get("task_group", "unknown")
        by_group[g].append(ep)
    keys = sorted(by_group.keys())
    chosen: list[int] = []
    while len(chosen) < num and keys:
        for k in list(keys):
            if not by_group[k]:
                keys.remove(k); continue
            chosen.append(by_group[k].pop(0))
            if len(chosen) >= num:
                break
    return chosen[:num]


# ============================================================================
# Map lookup helpers
# ============================================================================


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


def _resize_3d_map(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a (T, H, W) map to target_shape (best-effort). 2D maps are
    promoted; mismatched T axes are averaged or broadcast."""
    import cv2
    Tt, Ht, Wt = target_shape
    a = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    if a.ndim == 2:
        r = cv2.resize(a, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        return np.broadcast_to(r[None, :, :], (Tt, Ht, Wt)).copy()
    if a.ndim != 3:
        return np.zeros(target_shape, dtype=np.float32)
    if a.shape == (Tt, Ht, Wt):
        return a
    # Handle T mismatch by averaging if too many time slices, broadcasting if too few.
    if a.shape[0] != Tt:
        a = np.nanmean(a, axis=0, keepdims=True)
        a = np.broadcast_to(a, (Tt, *a.shape[1:])).copy()
    out = np.stack([
        cv2.resize(a[t], (Wt, Ht), interpolation=cv2.INTER_LINEAR) for t in range(Tt)
    ], axis=0)
    return out


# ============================================================================
# Metrics
# ============================================================================


def _action_metrics(base_chunk: np.ndarray, pert_chunk: np.ndarray,
                    gt_chunk: np.ndarray | None) -> dict[str, float]:
    out: dict[str, float] = {
        "action_l2": float("nan"), "action_first": float("nan"),
        "action_near": float("nan"), "action_far": float("nan"),
        "action_gripper": float("nan"), "action_cosine": float("nan"),
        "action_l2_vs_gt": float("nan"),
    }
    if base_chunk.shape == pert_chunk.shape and base_chunk.size > 0:
        H, A = base_chunk.shape
        third = max(1, H // 3)
        diff = base_chunk - pert_chunk
        out["action_l2"] = float(np.sqrt((diff ** 2).mean()))
        out["action_first"] = float(np.sqrt((diff[:1] ** 2).mean()))
        out["action_near"] = float(np.sqrt((diff[:third] ** 2).mean()))
        out["action_far"] = float(np.sqrt((diff[-third:] ** 2).mean()))
        out["action_gripper"] = float(np.sqrt((diff[:, -1:] ** 2).mean()))
        a = base_chunk.flatten().astype(np.float64)
        b = pert_chunk.flatten().astype(np.float64)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na > 0 and nb > 0:
            out["action_cosine"] = float(1.0 - np.dot(a, b) / (na * nb))
    if gt_chunk is not None and gt_chunk.size > 0 and pert_chunk.size > 0:
        H = min(gt_chunk.shape[0], pert_chunk.shape[0])
        A = min(gt_chunk.shape[1], pert_chunk.shape[1])
        if H > 0 and A > 0:
            diff = gt_chunk[:H, :A] - pert_chunk[:H, :A]
            out["action_l2_vs_gt"] = float(np.sqrt((diff ** 2).mean()))
    return out


def _gripper_event(gt_chunk: np.ndarray | None) -> int:
    """1 if the GT action chunk contains a gripper-state transition (sign change)."""
    if gt_chunk is None or gt_chunk.size == 0:
        return 0
    g = np.asarray(gt_chunk[:, -1], dtype=np.float64)
    if g.size < 2:
        return 0
    # Treat positive vs non-positive as open/closed.
    s = (g > 0.5).astype(np.int8) if (g.max() > 1.0 or g.min() < -0.05) else (g > 0).astype(np.int8)
    return int(np.any(np.abs(np.diff(s)) >= 1))


def _retention_fraction(importance: np.ndarray, budget_pct: float) -> float:
    flat = np.nan_to_num(importance, nan=0.0).flatten()
    if flat.size == 0:
        return 0.0
    n = max(1, int(round(flat.size * budget_pct / 100.0)))
    n = min(n, flat.size)
    th = np.partition(flat, -n)[-n]
    return float((flat >= th).mean())


# ============================================================================
# Per-trajectory sweep
# ============================================================================


def sweep_trajectory(policy: GrootSimPolicy,
                     entries: list[dict[str, Any]],
                     methods: list[str],
                     budgets: list[float],
                     variants: list[str],
                     primary_op: str, primary_metric: str,
                     primary_video_metric: str,
                     hybrid_alpha: float,
                     maps_dir: Path | None,
                     saliency_dir: Path | None,
                     seed: int,
                     out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    # For variant C we may need a previous-frame ys per timestep.
    for step_idx, entry in enumerate(entries):
        try:
            obs = build_obs(entry)
        except Exception as e:
            logger.warning("[ep%s t%s] build_obs failed: %s",
                           entry.get("episode_index"), entry.get("frame_index"), e)
            continue

        try:
            baseline = ps.capture_baseline(policy, obs, seed)
        except Exception as e:
            logger.warning("[ep%s t%s] baseline capture failed: %s",
                           entry.get("episode_index"), entry.get("frame_index"), e)
            continue

        T_lat, H_lat, W_lat = (
            int(baseline.ys.shape[2]), int(baseline.ys.shape[3]), int(baseline.ys.shape[4])
        )
        latent_shape = (T_lat, H_lat, W_lat)

        gt_chunk = np.asarray(entry.get("gt_action_chunk", []), dtype=np.float64)
        gripper_evt = _gripper_event(gt_chunk if gt_chunk.size else None)

        stage1 = _load_stage1_arrays(maps_dir, entry["example_id"])
        sal = _load_saliency_arrays(saliency_dir, entry["example_id"])

        # If the per-step Stage-1 map is missing, fall back to the closest neighbour
        # in the same episode (so action_causal still works on those steps).
        if stage1 is None and maps_dir is not None:
            for nb in entries:
                cand = _load_stage1_arrays(maps_dir, nb["example_id"])
                if cand is not None:
                    stage1 = cand; break

        # Build importance per method
        importance_per_method: dict[str, np.ndarray] = {}
        for m in methods:
            imp = _load_method_map(
                m, latent_shape, stage1, sal, hybrid_alpha,
                primary_op, primary_metric, primary_video_metric, rng,
            )
            if imp is None:
                continue
            if imp.shape != latent_shape:
                try:
                    imp = _resize_3d_map(imp, latent_shape)
                except Exception:
                    continue
            importance_per_method[m] = imp.astype(np.float32)

        prev_ys: torch.Tensor | None = None
        if "C_temporal_cache" in variants:
            prev_ys = ps.fetch_prev_frame_ys(policy, entry, lag=1, seed=seed)

        for method, imp in importance_per_method.items():
            for budget in budgets:
                for variant in variants:
                    if variant == "C_temporal_cache" and prev_ys is None and budget < 100.0 - 1e-6:
                        continue
                    try:
                        ys_pert = _build_perturbed_ys(
                            variant, baseline.ys.clone(), imp, budget,
                            baseline.z_channel_start, prev_ys=prev_ys,
                        )
                    except Exception as e:
                        logger.warning("[ep%s t%s] %s/%s/%g build failed: %s",
                                       entry.get("episode_index"), entry.get("frame_index"),
                                       method, variant, budget, e)
                        continue
                    try:
                        pert_chunk, _pert_actions, pert_video, fwd_s = ps.run_perturbed(
                            policy, baseline, ys_pert
                        )
                    except Exception as e:
                        logger.warning("[ep%s t%s] %s/%s/%g forward failed: %s",
                                       entry.get("episode_index"), entry.get("frame_index"),
                                       method, variant, budget, e)
                        continue
                    am = _action_metrics(baseline.action_chunk, pert_chunk,
                                         gt_chunk if gt_chunk.size else None)
                    retain = _retention_fraction(imp, budget)
                    rows.append({
                        "example_id": entry["example_id"],
                        "episode_index": int(entry.get("episode_index", -1)),
                        "frame_index": int(entry.get("frame_index", -1)),
                        "step_within_episode": step_idx,
                        "role": entry.get("role"),
                        "task_group": entry.get("task_group"),
                        "instruction": (entry.get("instruction") or "")[:120],
                        "method": method,
                        "variant": variant,
                        "budget_pct": float(budget),
                        "fraction_retained_actual": retain,
                        "approx_latency_ratio": float(budget) / 100.0,
                        "forward_elapsed_s": float(fwd_s),
                        **am,
                        "gripper_event": int(gripper_evt),
                        "valid": int(not (math.isnan(am["action_l2"])
                                          or np.any(np.isnan(pert_chunk)))),
                    })

    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with (out_dir / "rows.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True)
    p.add_argument("--maps_dir", required=True)
    p.add_argument("--saliency_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_episodes", type=int, default=6)
    p.add_argument("--group_balanced", action="store_true",
                   help="Spread chosen episodes across task_groups.")
    p.add_argument("--budgets", default="100,75,50,25")
    p.add_argument("--methods",
                   default="action_causal,cala_wam_hybrid,uniform,attention,gripper,object,flow,random")
    p.add_argument("--variants", default="A_hard_retention",
                   help="Comma-separated. Only A_hard_retention by default for clarity.")
    p.add_argument("--hybrid_alpha", type=float, default=0.7)
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
    by_ep = group_into_trajectories(manifest)
    chosen_eps = _select_episodes(by_ep, args.num_episodes, args.group_balanced)
    if not chosen_eps:
        logger.error("No episodes in manifest."); sys.exit(1)
    logger.info("Selected %d episodes: %s", len(chosen_eps), chosen_eps)

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
    for i, ep_idx in enumerate(chosen_eps):
        ep_entries = by_ep[ep_idx]
        ep_dir = out_dir / f"episode_{ep_idx:06d}"
        if args.skip_existing and (ep_dir / "rows.csv").exists():
            logger.info("[%d/%d] ep %s already done, skipping", i + 1, len(chosen_eps), ep_idx)
            with (ep_dir / "rows.csv").open("r") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    all_rows.append({k: _coerce(v) for k, v in row.items()})
            continue
        logger.info("[%d/%d] sweeping episode %d (%d steps, %d methods × %d budgets × %d variants)",
                    i + 1, len(chosen_eps), ep_idx, len(ep_entries),
                    len(methods), len(budgets), len(variants))
        t0 = time.perf_counter()
        rows = sweep_trajectory(
            policy=policy, entries=ep_entries, methods=methods, budgets=budgets,
            variants=variants,
            primary_op=args.primary_operator, primary_metric=args.primary_metric,
            primary_video_metric=args.primary_video_metric,
            hybrid_alpha=args.hybrid_alpha,
            maps_dir=maps_dir, saliency_dir=sal_dir,
            seed=args.seed, out_dir=ep_dir,
        )
        all_rows.extend(rows)
        logger.info("[%d/%d] ep %s done in %.0fs (%d rows)",
                    i + 1, len(chosen_eps), ep_idx,
                    time.perf_counter() - t0, len(rows))

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
        "n_episodes": len(chosen_eps),
        "episodes": chosen_eps,
        "methods": methods, "budgets": budgets, "variants": variants,
        "n_rows": len(all_rows),
    }, indent=2))
    logger.info("Done. all_rows.csv -> %s", out_dir / "all_rows.csv")


if __name__ == "__main__":
    main()
