#!/usr/bin/env python3
"""Stage 1 / Compute — Per-block causal-importance maps for DreamZero-DROID.

Implements the **core perturbation pass** that Stage-1 analysis (E1.1 – E1.5)
consumes. For each manifest example we sweep every (T_lat, H_lat, W_lat)
block of the action head's VAE-encoded conditioning latent `head.ys` and,
for each Stage-1 perturbation operator, measure how much the predicted
action chunk and the generated future-video latents change. Output is a set
of per-example heatmap arrays + a baseline-input image, written into a
deterministic directory layout that the analysis script reads.

For a given example with latent grid `(T_lat, H_lat, W_lat)`, an operator
`M`, and block index `i = (t, h, w)` of size `(bt, bh, bw)`:

    s_i^M = ||a(z) - a(M_i z)||             (action sensitivity)
    v_i^M = ||v(z) - v(M_i z)||             (future-latent sensitivity)
    g_i^M = | gripper(a(z)) - gripper(a(M_i z)) |
    f_i^M = ||a_first(z) - a_first(M_i z)|| (near-horizon sensitivity)

We also record `perturbation_norm_relative_i^M = ||z_i' - z_i|| / ||z_i||`
so the analysis script can produce both raw and norm-normalised heatmaps.

This script is **GPU heavy** — it does `num_blocks * num_operators` forward
passes per example. Use `--block_size N` (default 1) for a coarser grid,
`--num_examples K` and `--max_blocks B` to bound runtime, and reduce
`--operators` to fewer choices for fast iteration.

Example (full pass, ~1.5 h for 8 examples on H100):

    python scripts/stage1/causal_maps_compute.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_examples 8 \\
        --operators local_mean,blur_k3,prev_frame_cache_l1,zero_mask \\
        --alpha 1.0 \\
        --block_size 1 \\
        --output_dir runs/stage1_maps

Smoke (fast) example (~6 min):

    python scripts/stage1/causal_maps_compute.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_examples 2 --operators local_mean --block_size 2 \\
        --output_dir runs/stage1_maps_smoke

Per-example output layout:

    runs/stage1_maps/<example_id>/
        meta.json                       — example metadata + latent shape
        baseline_input.png              — first camera view at frame_index
        baseline_decoded.png            — VAE-decoded baseline ys (sanity)
        heatmaps.npz                    — operator × metric × (T,H,W) arrays
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage1.compute")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

# Local imports — Stage 0 helpers and operators
SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
sys.path.insert(0, str(STAGE0_DIR))

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

from _common import (  # noqa: E402
    EMBODIMENT_TAG,
    build_obs,
    init_dist_single_process,
    read_frame,
    reset_causal_state,
)

import perturbation_suite as ps  # noqa: E402  (for capture_baseline / run_perturbed / op_*)


# ============================================================================
# Operator catalogue used by Stage 1 (subset of Stage-0 perturbation_suite)
# ============================================================================


STAGE1_OPERATORS: dict[str, dict[str, Any]] = {
    "local_mean":          {"fn": ps.op_local_mean,         "category": "in_distribution"},
    "frame_mean":          {"fn": ps.op_frame_mean,         "category": "in_distribution"},
    "dataset_mean":        {"fn": ps.op_dataset_mean,       "category": "in_distribution",
                            "needs": ["dataset_mean_lat"]},
    "blur_k3":             {"fn": ps.op_blur,               "category": "compression",
                            "extras": {"kernel_size": 3}},
    "blur_k5":             {"fn": ps.op_blur,               "category": "compression",
                            "extras": {"kernel_size": 5}},
    "avg_pool_2":          {"fn": ps.op_avg_pool,           "category": "compression",
                            "extras": {"factor": 2}},
    "avg_pool_4":          {"fn": ps.op_avg_pool,           "category": "compression",
                            "extras": {"factor": 4}},
    "zero_mask":           {"fn": ps.op_zero_mask,          "category": "destructive"},
    "gaussian_noise":      {"fn": ps.op_gaussian_noise,     "category": "noise"},
    "prev_frame_cache_l1": {"fn": ps.op_prev_frame_cache,   "category": "temporal",
                            "needs": ["prev_ys_l1"]},
    "prev_frame_cache_l2": {"fn": ps.op_prev_frame_cache,   "category": "temporal",
                            "needs": ["prev_ys_l2"]},
    "prev_frame_cache_l4": {"fn": ps.op_prev_frame_cache,   "category": "temporal",
                            "needs": ["prev_ys_l4"]},
}


# ============================================================================
# Block iterator
# ============================================================================


def _iter_blocks(latent_shape: tuple[int, int, int],
                 block_size: tuple[int, int, int]) -> Iterable[tuple[tuple[slice, slice, slice], tuple[int, int, int]]]:
    """Yield (region_slices, block_index) covering the latent grid.

    block_index is the integer position in the down-sampled heatmap grid.
    """
    T, H, W = latent_shape
    bt, bh, bw = block_size
    bt = max(1, min(bt, T))
    bh = max(1, min(bh, H))
    bw = max(1, min(bw, W))
    nT = math.ceil(T / bt)
    nH = math.ceil(H / bh)
    nW = math.ceil(W / bw)
    for it in range(nT):
        for ih in range(nH):
            for iw in range(nW):
                t0 = it * bt; t1 = min(T, t0 + bt)
                h0 = ih * bh; h1 = min(H, h0 + bh)
                w0 = iw * bw; w1 = min(W, w0 + bw)
                yield (slice(t0, t1), slice(h0, h1), slice(w0, w1)), (it, ih, iw)


# ============================================================================
# Metric extraction
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


def _latent_perturbation_norm(ys_base: torch.Tensor, ys_pert: torch.Tensor,
                              region: tuple[slice, slice, slice], ch0: int) -> float:
    sl_t, sl_h, sl_w = region
    block_orig = ys_base[:, ch0:, sl_t, sl_h, sl_w].float()
    block_pert = ys_pert[:, ch0:, sl_t, sl_h, sl_w].float()
    base_norm = float(block_orig.norm().clamp_min(1e-8).item())
    diff_norm = float((block_pert - block_orig).norm().item())
    return diff_norm / base_norm


# ============================================================================
# Visualisation helpers
# ============================================================================


def _save_input_image(entry: dict[str, Any], out_path: Path,
                      camera: str = "video.exterior_image_1_left") -> bool:
    try:
        import cv2
        path = (entry.get("video_paths") or {}).get(camera)
        if not path:
            paths = list((entry.get("video_paths") or {}).values())
            path = paths[0] if paths else None
        if not path:
            return False
        rgb = read_frame(path, int(entry.get("frame_index", 0)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        logger.warning("Save input image failed: %s", e)
        return False


def _save_decoded_baseline(head: Any, ys_base: torch.Tensor, out_path: Path) -> bool:
    """Decode the first time-slice of baseline ys to RGB for sanity."""
    try:
        import cv2
        z = ys_base[:, 4:, :, :, :].float()
        with torch.inference_mode():
            params = next(head.vae.parameters(), None)
            if params is None:
                return False
            z = z.to(params.device, dtype=params.dtype)
            frames = head.vae.decode(
                z,
                tiled=getattr(head, "tiled", True),
                tile_size=(getattr(head, "tile_size_height", 34), getattr(head, "tile_size_width", 34)),
                tile_stride=(getattr(head, "tile_stride_height", 18), getattr(head, "tile_stride_width", 16)),
            )
        f0 = frames[0, :, 0]
        f0 = ((f0.float() + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        rgb = np.transpose(f0, (1, 2, 0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        logger.debug("Decoded baseline save failed: %s", e)
        return False


# ============================================================================
# Per-example causal map
# ============================================================================


def compute_example(policy: GrootSimPolicy,
                    entry: dict[str, Any],
                    operators: list[str],
                    alpha: float,
                    block_size: tuple[int, int, int],
                    seed: int,
                    max_blocks: int | None,
                    out_dir: Path,
                    dataset_mean_lat: torch.Tensor | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head

    # Baseline (capture ys, action, video_pred)
    obs = build_obs(entry)
    baseline = ps.capture_baseline(policy, obs, seed)
    T, H, W = int(baseline.ys.shape[2]), int(baseline.ys.shape[3]), int(baseline.ys.shape[4])
    logger.info("[%s] latent shape T×H×W = %d × %d × %d", entry["example_id"], T, H, W)

    # Auxiliary inputs lazily computed when needed
    aux_cache: dict[str, Any] = {}

    def _aux(name: str) -> Any:
        if name == "dataset_mean_lat":
            return dataset_mean_lat
        if name in aux_cache:
            return aux_cache[name]
        if name.startswith("prev_ys_l"):
            lag = int(name.split("l")[-1])
            aux_cache[name] = ps.fetch_prev_frame_ys(policy, entry, lag, seed)
            return aux_cache[name]
        return None

    # Allocate heatmaps per (operator, metric)
    bt, bh, bw = block_size
    nT = math.ceil(T / bt); nH = math.ceil(H / bh); nW = math.ceil(W / bw)
    metrics_keys = ("action_l2", "action_first", "action_near", "action_far",
                    "action_gripper", "action_cosine", "video_l2",
                    "perturbation_norm_relative")

    heatmaps: dict[str, dict[str, np.ndarray]] = {
        op_name: {m: np.full((nT, nH, nW), np.nan, dtype=np.float32) for m in metrics_keys}
        for op_name in operators
    }
    valid_blocks: dict[str, np.ndarray] = {
        op_name: np.zeros((nT, nH, nW), dtype=np.int8) for op_name in operators
    }

    # Save sanity images
    _save_input_image(entry, out_dir / "baseline_input.png")
    _save_decoded_baseline(head, baseline.ys, out_dir / "baseline_decoded.png")

    rng = np.random.default_rng(seed)

    # Filter operators that need missing inputs (auto-skip)
    feasible_operators: list[str] = []
    for op_name in operators:
        spec = STAGE1_OPERATORS.get(op_name)
        if spec is None:
            logger.warning("Unknown operator %s; skipping.", op_name); continue
        ok = True
        for need in spec.get("needs", []):
            if _aux(need) is None:
                ok = False
                logger.info("Operator %s requires %s which is unavailable; skipping.", op_name, need)
                break
        if ok:
            feasible_operators.append(op_name)

    if not feasible_operators:
        logger.warning("[%s] no feasible operators; skipping.", entry["example_id"])
        return {"example_id": entry["example_id"], "skipped": True}

    blocks = list(_iter_blocks((T, H, W), (bt, bh, bw)))
    if max_blocks is not None and len(blocks) > max_blocks:
        idxs = rng.choice(len(blocks), size=max_blocks, replace=False)
        blocks = [blocks[i] for i in sorted(idxs)]
    n_blocks = len(blocks)
    logger.info("[%s] sweeping %d blocks × %d operators = %d forward passes",
                entry["example_id"], n_blocks, len(feasible_operators),
                n_blocks * len(feasible_operators))

    t_start = time.perf_counter()
    for bi, (region, idx) in enumerate(blocks):
        for op_name in feasible_operators:
            spec = STAGE1_OPERATORS[op_name]
            kwargs: dict[str, Any] = dict(spec.get("extras", {}) or {})
            for need in spec.get("needs", []):
                kwargs[need.replace("dataset_mean_lat", "dataset_mean_lat")] = _aux(need)
            kwargs["rng"] = rng
            try:
                ys_pert = spec["fn"](baseline.ys.clone(), region, baseline.z_channel_start,
                                     strength=alpha, **kwargs)
            except Exception as e:
                logger.warning("op %s build failed at block %s: %s", op_name, idx, e)
                continue
            try:
                pert_chunk, _pert_actions, pert_video, _ = ps.run_perturbed(policy, baseline, ys_pert)
            except Exception as e:
                logger.warning("op %s forward failed at block %s: %s", op_name, idx, e)
                continue

            am = _action_metrics(baseline.action_chunk, pert_chunk)
            v_l2 = _video_l2(baseline.video_pred, pert_video)
            pn = _latent_perturbation_norm(baseline.ys, ys_pert, region, baseline.z_channel_start)

            t_idx, h_idx, w_idx = idx
            heatmaps[op_name]["action_l2"][t_idx, h_idx, w_idx] = am["action_l2"]
            heatmaps[op_name]["action_first"][t_idx, h_idx, w_idx] = am["action_first"]
            heatmaps[op_name]["action_near"][t_idx, h_idx, w_idx] = am["action_near"]
            heatmaps[op_name]["action_far"][t_idx, h_idx, w_idx] = am["action_far"]
            heatmaps[op_name]["action_gripper"][t_idx, h_idx, w_idx] = am["action_gripper"]
            heatmaps[op_name]["action_cosine"][t_idx, h_idx, w_idx] = am["action_cosine"]
            heatmaps[op_name]["video_l2"][t_idx, h_idx, w_idx] = v_l2
            heatmaps[op_name]["perturbation_norm_relative"][t_idx, h_idx, w_idx] = pn
            valid_blocks[op_name][t_idx, h_idx, w_idx] = 1

        if (bi + 1) % max(1, n_blocks // 10) == 0 or bi == n_blocks - 1:
            elapsed = time.perf_counter() - t_start
            done = bi + 1
            eta = elapsed * (n_blocks - done) / max(done, 1)
            logger.info("[%s] block %d/%d (%.0f%%)  elapsed=%.0fs  eta=%.0fs",
                        entry["example_id"], done, n_blocks, 100.0 * done / n_blocks,
                        elapsed, eta)

    total = time.perf_counter() - t_start

    # Save artifacts
    np_payload = {}
    for op_name in feasible_operators:
        for m in metrics_keys:
            np_payload[f"{op_name}__{m}"] = heatmaps[op_name][m]
        np_payload[f"{op_name}__valid"] = valid_blocks[op_name]
    np.savez_compressed(out_dir / "heatmaps.npz", **np_payload)

    # Baseline action chunk + video summary (helps later analysis without re-running)
    np.save(out_dir / "baseline_action_chunk.npy", baseline.action_chunk)

    meta = {
        "example_id": entry["example_id"],
        "task_group": entry.get("task_group"),
        "task_name": entry.get("task_name"),
        "instruction": entry.get("instruction"),
        "episode_index": entry.get("episode_index"),
        "frame_index": entry.get("frame_index"),
        "role": entry.get("role"),
        "video_paths": entry.get("video_paths"),
        "operators_used": feasible_operators,
        "alpha": float(alpha),
        "block_size": [int(bt), int(bh), int(bw)],
        "latent_shape": [T, H, W],
        "heatmap_shape": [int(nT), int(nH), int(nW)],
        "z_channel_start": int(baseline.z_channel_start),
        "n_blocks": int(n_blocks),
        "elapsed_s": float(total),
        "seed": int(seed),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("[%s] done in %.1f min, saved to %s", entry["example_id"], total / 60, out_dir)
    return meta


# ============================================================================
# Driver
# ============================================================================


def _select_examples(manifest: list[dict[str, Any]], num: int,
                     phase_balanced: bool) -> list[dict[str, Any]]:
    if not phase_balanced:
        return manifest[:num]
    # Stratify by (task_group, role) so phase-analysis later has enough data.
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


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_examples", type=int, default=8)
    p.add_argument("--phase_balanced", action="store_true",
                   help="Stratify selected examples across (task_group, role) for E1.3 phase analysis.")
    p.add_argument("--operators", default="local_mean,blur_k3,prev_frame_cache_l1,zero_mask",
                   help=f"Comma-separated subset of {sorted(STAGE1_OPERATORS.keys())}.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Perturbation strength for interpolation operators.")
    p.add_argument("--block_size", default="1x1x1",
                   help="Block size in latent space, T_xH_xW_ (e.g. 1x2x2).")
    p.add_argument("--max_blocks", type=int, default=None,
                   help="Cap blocks per example (random subset). Default: full grid.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip examples whose output_dir already has heatmaps.npz.")
    args = p.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bs_tokens = args.block_size.lower().split("x")
    if len(bs_tokens) != 3:
        raise ValueError("--block_size must be TxHxW (e.g. 1x2x2)")
    block_size = (int(bs_tokens[0]), int(bs_tokens[1]), int(bs_tokens[2]))

    operators = [o.strip() for o in args.operators.split(",") if o.strip()]
    bad = [o for o in operators if o not in STAGE1_OPERATORS]
    if bad:
        raise ValueError(f"Unknown operators: {bad}. Available: {sorted(STAGE1_OPERATORS.keys())}")

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

    # Pre-pass: capture baselines to compute dataset_mean_lat (only if needed).
    needs_dataset_mean = any(
        "dataset_mean_lat" in (STAGE1_OPERATORS[o].get("needs", []) or []) for o in operators
    )
    dataset_mean_lat: torch.Tensor | None = None
    if needs_dataset_mean:
        logger.info("Computing dataset_mean_lat across %d examples ...", len(chosen))
        baselines: list[ps.BaselineCache] = []
        for entry in chosen:
            try:
                obs = build_obs(entry)
                baselines.append(ps.capture_baseline(policy, obs, args.seed))
            except Exception as e:
                logger.warning("baseline capture failed for %s: %s", entry.get("example_id"), e)
        if baselines:
            dataset_mean_lat = ps.compute_dataset_mean(baselines)
            logger.info("dataset_mean_lat shape=%s", list(dataset_mean_lat.shape) if dataset_mean_lat is not None else None)

    # Per-example sweep
    all_meta: list[dict[str, Any]] = []
    for i, entry in enumerate(chosen):
        ex_dir = out_dir / entry["example_id"]
        if args.skip_existing and (ex_dir / "heatmaps.npz").exists():
            logger.info("[%d/%d] skipping %s (already computed)", i + 1, len(chosen), entry["example_id"])
            try:
                all_meta.append(json.loads((ex_dir / "meta.json").read_text()))
            except Exception:
                pass
            continue
        logger.info("[%d/%d] computing causal map for %s ...", i + 1, len(chosen), entry["example_id"])
        try:
            meta = compute_example(
                policy=policy, entry=entry, operators=operators, alpha=args.alpha,
                block_size=block_size, seed=args.seed, max_blocks=args.max_blocks,
                out_dir=ex_dir, dataset_mean_lat=dataset_mean_lat,
            )
            all_meta.append(meta)
        except Exception as e:
            logger.error("compute_example failed for %s: %s", entry["example_id"], e, exc_info=True)
            continue

    # Run-level manifest (for the analysis script)
    (out_dir / "compute_run.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "operators": operators,
        "alpha": args.alpha,
        "block_size": list(block_size),
        "seed": args.seed,
        "num_examples_completed": len([m for m in all_meta if not m.get("skipped")]),
        "num_examples_requested": len(chosen),
        "examples": all_meta,
    }, indent=2))
    logger.info("All examples done. Manifest -> %s", out_dir / "compute_run.json")


if __name__ == "__main__":
    main()
