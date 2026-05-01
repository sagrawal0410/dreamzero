#!/usr/bin/env python3
"""Stage 0 / E0.1 — Baseline inference sanity for DreamZero-DROID.

Loads the manifest produced by `scripts/stage0/build_task_suite.py`, runs
unmodified DreamZero (no perturbation) on every example, and writes the
artifacts the rest of Stage 0 needs:

    Per-example  : input observations, instructions, proprio,
                   predicted action chunks, generated future video
                   (latents + decoded MP4 if requested), DiT/VAE latents,
                   per-stage timing, peak GPU memory, NaN/range checks.
    Stability    : per-example action-chunk variance across `--num_seeds`
                   runs (same input, different noise), latency variance,
                   video-latent variance.
    Aggregate    : summary.json with means/medians/quantiles broken down
                   by task group, plus a baseline_table.csv suitable for
                   pasting into a paper appendix.
    Latent map   : latent_map.json documenting the exact tensor shapes,
                   spatial / temporal resolution, channel dim, and what
                   each latent corresponds to (action vs future video,
                   pre- vs post-VAE).

Example:

    python scripts/stage0/eval_baseline.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_seeds 3 \\
        --save_latents \\
        --save_actions \\
        --save_videos \\
        --profile_latency \\
        --output_dir runs/stage0_baseline

The script runs as a single process. GrootSimPolicy initialises a 1-rank
process group (gloo) automatically. For multi-GPU model-parallel inference,
use the WebSocket server entrypoint instead — this script is intentionally
the offline, deterministic, debuggable path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage0.eval_baseline")

# Disable torch dynamo to avoid recompile churn on varying batch shapes.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402  (must come after env var)
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

import torch.distributed as dist  # noqa: E402
from tianshou.data import Batch  # noqa: E402

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402


# ----------------------------------------------------------------------------
# Default modality keys for OXE_DROID — must match training config
# ----------------------------------------------------------------------------

VIDEO_KEYS = (
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
)
LANGUAGE_KEY = "annotation.language.action_text"
ACTION_KEYS = ("action.joint_position", "action.gripper_position")
EMBODIMENT_TAG = "oxe_droid"


# ----------------------------------------------------------------------------
# Distributed init for single-process run
# ----------------------------------------------------------------------------


def _init_dist_single_process() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


# ----------------------------------------------------------------------------
# Frame loading
# ----------------------------------------------------------------------------


def _read_frame(mp4_path: str, frame_index: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {mp4_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _build_obs(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a single-frame observation dict in the format GrootSimPolicy expects.

    For the AR_droid causal head the first call uses 1 frame (T=1, H, W, 3).
    State is (1, D) shaped per training config.
    """
    obs: dict[str, Any] = {}
    for key in VIDEO_KEYS:
        mp4 = entry["video_paths"].get(key)
        if mp4 is None:
            raise KeyError(f"Manifest entry missing video for {key}: {entry['example_id']}")
        frame = _read_frame(mp4, entry["frame_index"])
        obs[key] = frame[np.newaxis, ...].astype(np.uint8)  # (1, H, W, 3)

    state_vec = np.asarray(entry["state_at_t"], dtype=np.float64).reshape(-1)
    # DROID convention: 7 joints + 1 gripper if state is 8-dim, else split last as gripper.
    if state_vec.size >= 8:
        obs["state.joint_position"] = state_vec[:7].reshape(1, -1).astype(np.float64)
        obs["state.gripper_position"] = state_vec[7:8].reshape(1, -1).astype(np.float64)
    else:
        obs["state.joint_position"] = state_vec[:7].reshape(1, -1).astype(np.float64)
        obs["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

    obs[LANGUAGE_KEY] = entry.get("instruction", "") or ""
    return obs


# ----------------------------------------------------------------------------
# Policy reset between examples — clear causal KV cache so frame index resets
# ----------------------------------------------------------------------------


def _reset_causal_state(policy: GrootSimPolicy) -> None:
    head = getattr(policy.trained_model, "action_head", None)
    if head is None:
        return
    if hasattr(head, "current_start_frame"):
        head.current_start_frame = 0
    if hasattr(head, "language"):
        head.language = None
    if hasattr(head, "clip_feas"):
        head.clip_feas = None
    if hasattr(head, "ys"):
        head.ys = None


# ----------------------------------------------------------------------------
# One inference pass — returns predicted actions and timing
# ----------------------------------------------------------------------------


@dataclass
class InferResult:
    actions: dict[str, np.ndarray]
    action_chunk: np.ndarray             # (H, A) flat per-step action
    video_pred: torch.Tensor | None      # latent video tensor (kept on CPU)
    elapsed_s: float
    peak_gpu_mb: float
    has_nan: bool
    action_min: float
    action_max: float


def _flatten_action(actions: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for key in ACTION_KEYS:
        if key in actions:
            v = np.asarray(actions[key])
            if v.ndim == 1:
                v = v[None, :]
            parts.append(v)
    if not parts:
        return np.zeros((0, 0))
    H = max(p.shape[-2] if p.ndim >= 2 else 1 for p in parts)
    aligned = []
    for p in parts:
        if p.ndim == 1:
            p = np.tile(p[None, :], (H, 1))
        elif p.shape[-2] != H:
            reps = max(1, H // max(1, p.shape[-2]))
            p = np.tile(p, (reps, 1))[:H]
        aligned.append(p)
    return np.concatenate(aligned, axis=-1)


def _run_once(policy: GrootSimPolicy, obs: dict[str, Any], seed: int) -> InferResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    _reset_causal_state(policy)

    t0 = time.perf_counter()
    with torch.inference_mode():
        result_batch, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    actions: dict[str, np.ndarray] = {}
    for k in dir(result_batch.act):
        if k.startswith("action."):
            v = getattr(result_batch.act, k)
            if torch.is_tensor(v):
                v = v.detach().cpu().float().numpy()
            actions[k] = np.asarray(v)
    chunk = _flatten_action(actions)
    has_nan = bool(np.any(np.isnan(chunk))) if chunk.size else False
    a_min = float(np.min(chunk)) if chunk.size else 0.0
    a_max = float(np.max(chunk)) if chunk.size else 0.0

    if torch.is_tensor(video_pred):
        video_pred = video_pred.detach().to("cpu")

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() else 0.0
    )

    return InferResult(
        actions=actions,
        action_chunk=chunk,
        video_pred=video_pred,
        elapsed_s=elapsed,
        peak_gpu_mb=peak_mb,
        has_nan=has_nan,
        action_min=a_min,
        action_max=a_max,
    )


# ----------------------------------------------------------------------------
# Optional: decode latent video to MP4 for visual sanity
# ----------------------------------------------------------------------------


def _decode_video(policy: GrootSimPolicy, video_pred: torch.Tensor, out_path: Path) -> bool:
    try:
        import imageio
        from einops import rearrange
    except Exception:
        return False

    head = getattr(policy.trained_model, "action_head", None)
    if head is None or not hasattr(head, "vae"):
        return False

    try:
        with torch.inference_mode():
            vp = video_pred.to(next(head.vae.parameters()).device, dtype=next(head.vae.parameters()).dtype)
            frames = head.vae.decode(
                vp,
                tiled=getattr(head, "tiled", True),
                tile_size=(getattr(head, "tile_size_height", 34), getattr(head, "tile_size_width", 34)),
                tile_stride=(getattr(head, "tile_stride_height", 18), getattr(head, "tile_stride_width", 16)),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")
        frames = frames[0]
        frames = ((frames.float() + 1.0) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), list(frames), fps=5, codec="libx264")
        return True
    except Exception as e:
        logger.warning("Video decode failed for %s: %s", out_path.name, e)
        return False


# ----------------------------------------------------------------------------
# Aggregate stats helpers
# ----------------------------------------------------------------------------


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


# ----------------------------------------------------------------------------
# Main eval driver
# ----------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predicted_actions").mkdir(exist_ok=True)
    (out_dir / "latents").mkdir(exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "obs").mkdir(exist_ok=True)
    (out_dir / "per_example").mkdir(exist_ok=True)

    manifest_path = Path(args.task_suite).resolve()
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    logger.info("Loaded %d examples from %s", len(manifest), manifest_path)

    _init_dist_single_process()

    logger.info("Loading GrootSimPolicy from %s ...", args.checkpoint)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        model_path=args.checkpoint,
        device=args.device,
    )
    logger.info("Model loaded.")

    # Latent-map probe: run one example to discover tensor shapes.
    latent_map_written = False

    per_example_records: list[dict[str, Any]] = []
    latency_samples: list[float] = []
    peak_mem_samples: list[float] = []
    nan_count = 0

    # Per-task-group accumulators
    by_group: dict[str, list[dict[str, Any]]] = {}

    for i, entry in enumerate(manifest):
        ex_id = entry["example_id"]
        group = entry.get("task_group", "unknown")
        logger.info("[%d/%d] %s (group=%s)", i + 1, len(manifest), ex_id, group)

        try:
            obs = _build_obs(entry)
        except Exception as e:
            logger.warning("  build_obs failed (%s); skipping.", e)
            continue

        # Per-seed runs for stability measurement
        seeds = list(range(args.num_seeds))
        runs: list[InferResult] = []
        for s in seeds:
            try:
                runs.append(_run_once(policy, obs, seed=s))
            except Exception as e:
                logger.warning("  inference failed at seed=%d: %s", s, e)

        if not runs:
            continue

        # Stability metrics
        chunks = [r.action_chunk for r in runs if r.action_chunk.size]
        if len(chunks) >= 2 and all(c.shape == chunks[0].shape for c in chunks):
            stack = np.stack(chunks, axis=0)              # (S, H, A)
            action_std = float(np.mean(np.std(stack, axis=0)))
            action_max_abs_diff = float(np.max(np.abs(stack[0] - stack[-1])))
        else:
            action_std = 0.0
            action_max_abs_diff = 0.0

        latencies = [r.elapsed_s for r in runs]
        peak_mems = [r.peak_gpu_mb for r in runs]
        latency_samples.extend(latencies)
        peak_mem_samples.extend(peak_mems)
        if any(r.has_nan for r in runs):
            nan_count += 1

        # Save artifacts using the seed-0 run as canonical
        canonical = runs[0]

        if args.save_actions:
            np.savez(
                out_dir / "predicted_actions" / f"{ex_id}.npz",
                **{k: v for k, v in canonical.actions.items()},
                action_chunk_flat=canonical.action_chunk,
                gt_action_chunk=np.asarray(entry.get("gt_action_chunk", []), dtype=np.float64),
            )

        # Persist obs (downsampled — frames are uint8; this is small for 1 frame).
        if args.save_obs:
            np.savez(
                out_dir / "obs" / f"{ex_id}.npz",
                **{k: np.asarray(v) if not isinstance(v, str) else np.array([v]) for k, v in obs.items()},
            )

        if args.save_latents and canonical.video_pred is not None:
            torch.save(canonical.video_pred, out_dir / "latents" / f"{ex_id}__video_pred.pt")

        if args.save_videos and canonical.video_pred is not None:
            _decode_video(policy, canonical.video_pred, out_dir / "videos" / f"{ex_id}.mp4")

        # Latent-map artifact (write once)
        if not latent_map_written and canonical.video_pred is not None:
            latent_map = _build_latent_map(policy, canonical)
            (out_dir / "latent_map.json").write_text(json.dumps(latent_map, indent=2))
            latent_map_written = True

        # Compare to ground truth if action shapes line up
        gt_chunk_flat = np.asarray(entry.get("gt_action_chunk", []), dtype=np.float64)
        action_l1 = action_l2 = None
        if gt_chunk_flat.size and canonical.action_chunk.size:
            H = min(gt_chunk_flat.shape[0], canonical.action_chunk.shape[0])
            A = min(gt_chunk_flat.shape[1], canonical.action_chunk.shape[1])
            diff = canonical.action_chunk[:H, :A] - gt_chunk_flat[:H, :A]
            action_l1 = float(np.mean(np.abs(diff)))
            action_l2 = float(np.sqrt(np.mean(diff ** 2)))

        record = {
            "example_id": ex_id,
            "task_group": group,
            "task_name": entry.get("task_name"),
            "instruction": entry.get("instruction"),
            "episode_index": entry.get("episode_index"),
            "frame_index": entry.get("frame_index"),
            "role": entry.get("role"),
            "predicted_action_keys": list(canonical.actions.keys()),
            "predicted_action_shape_per_key": {
                k: list(v.shape) for k, v in canonical.actions.items()
            },
            "action_chunk_shape": list(canonical.action_chunk.shape),
            "action_min": canonical.action_min,
            "action_max": canonical.action_max,
            "has_nan": canonical.has_nan,
            "latency_s_per_seed": latencies,
            "latency_s_mean": float(np.mean(latencies)),
            "latency_s_std": float(np.std(latencies)) if len(latencies) > 1 else 0.0,
            "peak_gpu_mb_per_seed": peak_mems,
            "peak_gpu_mb_mean": float(np.mean(peak_mems)),
            "stability_action_std_mean": action_std,
            "stability_action_max_abs_diff": action_max_abs_diff,
            "open_loop_l1_vs_gt": action_l1,
            "open_loop_l2_vs_gt": action_l2,
            "video_pred_shape": (
                list(canonical.video_pred.shape)
                if canonical.video_pred is not None else None
            ),
            "saved": {
                "actions": bool(args.save_actions),
                "latents": bool(args.save_latents and canonical.video_pred is not None),
                "video": bool(args.save_videos and canonical.video_pred is not None),
                "obs": bool(args.save_obs),
            },
        }
        per_example_records.append(record)
        by_group.setdefault(group, []).append(record)

        (out_dir / "per_example" / f"{ex_id}.json").write_text(json.dumps(record, indent=2))

        logger.info(
            "  shape=%s  lat=%.3fs (std=%.3f)  peak=%.0fMB  std(act)=%.5f  l1=%s",
            record["action_chunk_shape"],
            record["latency_s_mean"],
            record["latency_s_std"],
            record["peak_gpu_mb_mean"],
            record["stability_action_std_mean"],
            f"{action_l1:.4f}" if action_l1 is not None else "n/a",
        )

    # Aggregate
    summary = {
        "checkpoint": args.checkpoint,
        "task_suite": str(manifest_path),
        "num_examples_total": len(manifest),
        "num_examples_evaluated": len(per_example_records),
        "num_seeds_per_example": args.num_seeds,
        "nan_count": nan_count,
        "latency_s": _summary_stats(latency_samples),
        "peak_gpu_mb": _summary_stats(peak_mem_samples),
        "stability_action_std": _summary_stats(
            [r["stability_action_std_mean"] for r in per_example_records]
        ),
        "open_loop_l1": _summary_stats(
            [r["open_loop_l1_vs_gt"] for r in per_example_records if r["open_loop_l1_vs_gt"] is not None]
        ),
        "open_loop_l2": _summary_stats(
            [r["open_loop_l2_vs_gt"] for r in per_example_records if r["open_loop_l2_vs_gt"] is not None]
        ),
        "by_group": {
            g: {
                "n": len(rs),
                "latency_s": _summary_stats([r["latency_s_mean"] for r in rs]),
                "open_loop_l1": _summary_stats(
                    [r["open_loop_l1_vs_gt"] for r in rs if r["open_loop_l1_vs_gt"] is not None]
                ),
                "stability_action_std": _summary_stats([r["stability_action_std_mean"] for r in rs]),
            }
            for g, rs in by_group.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_example.json").write_text(json.dumps(per_example_records, indent=2))

    # Paper-ready CSV
    with (out_dir / "baseline_table.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow([
            "example_id", "task_group", "instruction", "role",
            "latency_s_mean", "peak_gpu_mb_mean",
            "stability_action_std_mean", "open_loop_l1_vs_gt",
            "open_loop_l2_vs_gt", "has_nan",
        ])
        for r in per_example_records:
            w.writerow([
                r["example_id"], r["task_group"], (r["instruction"] or "")[:80],
                r["role"], f"{r['latency_s_mean']:.4f}", f"{r['peak_gpu_mb_mean']:.0f}",
                f"{r['stability_action_std_mean']:.6f}",
                "" if r["open_loop_l1_vs_gt"] is None else f"{r['open_loop_l1_vs_gt']:.6f}",
                "" if r["open_loop_l2_vs_gt"] is None else f"{r['open_loop_l2_vs_gt']:.6f}",
                int(bool(r["has_nan"])),
            ])

    logger.info("Wrote summary -> %s", out_dir / "summary.json")
    logger.info("Wrote per-example records -> %s", out_dir / "per_example.json")
    logger.info("Wrote baseline_table.csv -> %s", out_dir / "baseline_table.csv")
    logger.info("Latency p50=%.3fs  p90=%.3fs  peak GPU mean=%.0fMB  NaN=%d",
                summary["latency_s"].get("median", math.nan),
                summary["latency_s"].get("p90", math.nan),
                summary["peak_gpu_mb"].get("mean", math.nan),
                summary["nan_count"])


# ----------------------------------------------------------------------------
# Latent-map artifact
# ----------------------------------------------------------------------------


def _build_latent_map(policy: GrootSimPolicy, run: InferResult) -> dict[str, Any]:
    head = getattr(policy.trained_model, "action_head", None)
    cfg = getattr(head, "config", None) if head is not None else None
    map_: dict[str, Any] = {
        "video_pred_tensor": {
            "shape": list(run.video_pred.shape) if run.video_pred is not None else None,
            "dtype": str(run.video_pred.dtype) if run.video_pred is not None else None,
            "produced_by": "policy.trained_model.action_head.lazy_joint_video_action -> output.transpose(1, 2)",
            "stage": "DiT output latents (post causal denoising, pre-VAE-decode)",
            "decode_with": (
                "policy.trained_model.action_head.vae.decode("
                "tensor, tiled, tile_size, tile_stride) "
                "to recover RGB frames in [-1, 1]"
            ),
        },
        "action_chunk": {
            "shape": list(run.action_chunk.shape),
            "dtype": "float32",
            "produced_by": (
                "policy.unapply(...).act -> dict over ACTION_KEYS, "
                "flattened on the last dim"
            ),
            "normalized": False,
            "convention": (
                "DROID joint position (7) + gripper (1); per-horizon delta "
                "if relative_action enabled in train_cfg, else absolute."
            ),
        },
        "config": {
            "action_horizon": getattr(cfg, "action_horizon", None) if cfg else None,
            "num_inference_timesteps": getattr(cfg, "num_inference_timesteps", None) if cfg else None,
            "tiled": getattr(head, "tiled", None) if head else None,
            "tile_size": [
                getattr(head, "tile_size_height", None),
                getattr(head, "tile_size_width", None),
            ] if head else None,
            "tile_stride": [
                getattr(head, "tile_stride_height", None),
                getattr(head, "tile_stride_width", None),
            ] if head else None,
            "target_video_height": getattr(cfg, "target_video_height", None) if cfg else None,
            "target_video_width": getattr(cfg, "target_video_width", None) if cfg else None,
            "num_frame_per_block": getattr(head, "num_frame_per_block", None) if head else None,
        },
    }
    return map_


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to DreamZero checkpoint dir (config.json + model.safetensors + experiment_cfg/).")
    p.add_argument("--task_suite", required=True,
                   help="Path to manifest.json produced by build_task_suite.py.")
    p.add_argument("--output_dir", required=True,
                   help="Directory for artifacts (predicted_actions/, latents/, videos/, summary.json, ...).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for inference.")
    p.add_argument("--num_seeds", type=int, default=3,
                   help="Number of stability runs per example (different noise seeds).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of manifest entries to evaluate (for quick smoke runs).")
    p.add_argument("--save_actions", action="store_true", help="Save predicted action chunks as .npz per example.")
    p.add_argument("--save_latents", action="store_true", help="Save the DiT video latents (.pt) per example.")
    p.add_argument("--save_videos", action="store_true",
                   help="Decode video latents through the VAE and save MP4 per example.")
    p.add_argument("--save_obs", action="store_true",
                   help="Persist input observations (frames + state + prompt) per example.")
    p.add_argument("--profile_latency", action="store_true",
                   help="Reserved flag for compatibility with the planned interface; latency is always recorded.")
    args = p.parse_args()
    try:
        evaluate(args)
    except Exception as e:
        logger.error("eval_baseline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
