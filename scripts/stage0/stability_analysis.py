#!/usr/bin/env python3
"""Stage 0 / E0.3 — Baseline Action Stability for DreamZero-DROID.

Measures the **natural sampling variance** of unmodified DreamZero so that
later perturbation deltas can be judged against a real noise floor. Implements
all five Stage-0 stability sub-experiments and emits paper-ready artifacts:

    E0.3a — Same input, same seed (determinism check, expected ≈ 0).
    E0.3b — Same input, different seeds (the "noise floor").
    E0.3c — Diffusion-step ablation (1 / 2 / 4 / default DiT steps).
    E0.3d — Sampling-parameter sweep (sigma_shift if exposed).
    E0.3e — Per-horizon stability (early vs late action variance).

For each experiment we save:

    <out>/<exp>/per_example.csv         — long-form table, one row per run.
    <out>/<exp>/summary.json            — aggregate stats.
    <out>/<exp>/plot_*.png   plot_*.pdf — NeurIPS-style figures.

In addition we write:

    <out>/noise_floor.json    — per-horizon, per-dim seed-σ used as the
                                significance threshold for later stages.
    <out>/combined_report.md  — paper-appendix-ready summary with embedded
                                figure links and the headline numbers.

Example:

    python scripts/stage0/stability_analysis.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_examples 3 \\
        --num_seeds 16 \\
        --num_determinism_runs 8 \\
        --diffusion_steps 1,2,4,8 \\
        --sigma_shifts 3.0,5.0,7.0 \\
        --output_dir runs/stage0_stability

Notes
-----
* Runs as a single process. GrootSimPolicy initialises a 1-rank gloo group.
* The diffusion-step override mutates `head.num_inference_steps` and
  `head.dit_step_mask` for the duration of the run. We always restore them.
* Sigma-shift sweep is best-effort — it is applied only if the action head
  exposes `head.sigma_shift`. Otherwise E0.3d records `n/a`.
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
logger = logging.getLogger("stage0.stability")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

from tianshou.data import Batch  # noqa: E402

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

# Local helpers
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import (  # noqa: E402
    ACTION_KEYS,
    EMBODIMENT_TAG,
    build_obs,
    configure_neurips_matplotlib,
    flatten_action_dict,
    init_dist_single_process,
    reset_causal_state,
)


# ============================================================================
# Inference primitives
# ============================================================================


@dataclass
class RunResult:
    seed: int
    action_chunk: np.ndarray            # (H, A)
    actions_per_key: dict[str, np.ndarray]
    video_pred_summary: np.ndarray | None  # (T, C, H, W) summary stats only (mean/std per frame)
    elapsed_s: float
    has_nan: bool
    diffusion_steps_used: int
    sigma_shift_used: float | None


def _enable_determinism(deterministic: bool) -> None:
    """Enable / disable as much determinism as we can. Best-effort on CUDA."""
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = (not deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def _set_diffusion_steps(head: Any, n: int) -> tuple[int, list[bool]]:
    """Override num_inference_steps and dit_step_mask. Returns the originals."""
    orig_n = int(getattr(head, "num_inference_steps", 16))
    orig_mask = list(getattr(head, "dit_step_mask", [True] * orig_n))
    if n > 0:
        head.num_inference_steps = int(n)
        # Force every step to actually compute under the new schedule.
        head.dit_step_mask = [True] * int(n)
    return orig_n, orig_mask


def _restore_diffusion_steps(head: Any, orig_n: int, orig_mask: list[bool]) -> None:
    head.num_inference_steps = orig_n
    head.dit_step_mask = orig_mask


def _summarize_video(video_pred: torch.Tensor | None) -> np.ndarray | None:
    """Reduce a video-latent tensor to per-frame mean / std to keep memory low."""
    if video_pred is None or not torch.is_tensor(video_pred):
        return None
    try:
        # video_pred shape is typically (B, T, C, H, W)
        v = video_pred.detach().float().cpu()
        # Compress to (T, 2): mean and std per frame across spatial-channel dims
        if v.ndim >= 3:
            t_dim = 1 if v.ndim == 5 else 0
            keep_axes = list(range(v.ndim))
            keep_axes.pop(t_dim)
            if v.ndim == 5:
                keep_axes.pop(0)  # remove batch
            arr = v.numpy()
            # Move time to axis 0 then reduce
            if v.ndim == 5:
                arr = arr[0]                      # (T, C, H, W)
            mean = arr.mean(axis=tuple(range(1, arr.ndim)))
            std = arr.std(axis=tuple(range(1, arr.ndim)))
            return np.stack([mean, std], axis=-1)  # (T, 2)
    except Exception:
        return None
    return None


def _run(policy: GrootSimPolicy, obs: dict[str, Any], seed: int,
         deterministic: bool = False) -> RunResult:
    if deterministic:
        _enable_determinism(True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reset_causal_state(policy)

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
    chunk = flatten_action_dict(actions)
    has_nan = bool(np.any(np.isnan(chunk))) if chunk.size else False
    vp_summary = _summarize_video(video_pred)

    head = policy.trained_model.action_head
    return RunResult(
        seed=seed,
        action_chunk=chunk,
        actions_per_key=actions,
        video_pred_summary=vp_summary,
        elapsed_s=elapsed,
        has_nan=has_nan,
        diffusion_steps_used=int(getattr(head, "num_inference_steps", -1)),
        sigma_shift_used=float(getattr(head, "sigma_shift", float("nan"))),
    )


# ============================================================================
# Plotting helpers
# ============================================================================


def _save_fig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(str(base) + ext)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_e0_3a(per_run: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not per_run:
        return
    diffs = [r["max_abs_diff_action"] for r in per_run if r["max_abs_diff_action"] is not None]
    if not diffs:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(diffs)), diffs, color="#3b75af", edgecolor="black", linewidth=0.4)
    ax.axhline(1e-6, ls="--", lw=0.8, color="grey", label=r"$10^{-6}$ (numerical noise)")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("Run index")
    ax.set_ylabel(r"$\max_{i,h} |a_{r,i,h} - a_{0,i,h}|$")
    ax.set_title("E0.3a — Same input, same seed: max action divergence per run")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_max_abs_diff")


def _plot_e0_3b(seed_arrays_per_example: list[np.ndarray], out_dir: Path) -> None:
    """seed_arrays_per_example: list of (S, H, A) ndarrays."""
    import matplotlib.pyplot as plt
    valid = [a for a in seed_arrays_per_example if a.size and a.shape[0] >= 2]
    if not valid:
        return

    # Per-(horizon, dim) std averaged across examples after aligning shapes.
    H = min(a.shape[1] for a in valid)
    A = min(a.shape[2] for a in valid)
    cropped = [a[:, :H, :A] for a in valid]
    stds = np.stack([a.std(axis=0) for a in cropped], axis=0)  # (E, H, A)
    mean_std = stds.mean(axis=0)                                # (H, A)

    # Heatmap (H × A)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(mean_std.T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("Horizon step h")
    ax.set_ylabel("Action dim a")
    ax.set_title(r"E0.3b — Per-(horizon, dim) seed std $\sigma_{\mathrm{seed}}(h, a)$")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$\sigma_{\mathrm{seed}}$")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_std_heatmap")

    # Std vs horizon (averaged across dims; one line per example)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    cmap = plt.get_cmap("tab10")
    for i, a in enumerate(cropped):
        s = a.std(axis=0).mean(axis=-1)  # (H,)
        ax.plot(range(H), s, color=cmap(i % 10), alpha=0.85, label=f"ex {i}")
    avg = mean_std.mean(axis=-1)
    ax.plot(range(H), avg, color="black", lw=2.0, label="mean")
    ax.set_xlabel("Horizon step h")
    ax.set_ylabel(r"mean $\sigma_{\mathrm{seed}}$ across dims")
    ax.set_title("E0.3b — Seed variance vs horizon")
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_std_per_horizon")

    # Box plot of per-dim std distribution
    fig, ax = plt.subplots(figsize=(6, 3.2))
    per_dim = mean_std.mean(axis=0)  # (A,)
    box_data = [stds[..., a].flatten() for a in range(A)]
    ax.boxplot(box_data, showfliers=False, widths=0.6,
               medianprops=dict(color="#c14b4b", linewidth=1.2))
    ax.set_xticks(range(1, A + 1))
    ax.set_xticklabels([f"a{a}" for a in range(A)])
    ax.set_xlabel("Action dimension")
    ax.set_ylabel(r"$\sigma_{\mathrm{seed}}$")
    ax.set_title("E0.3b — Per-dim seed std (across horizons × examples)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_std_per_dim_box")

    # Action mean ± std vs horizon, per dim
    fig, axes = plt.subplots(2, math.ceil(A / 2), figsize=(2.6 * math.ceil(A / 2), 3.6),
                             squeeze=False, sharex=True)
    means = np.stack([a.mean(axis=0) for a in cropped], axis=0).mean(axis=0)  # (H, A)
    for a in range(A):
        ax = axes[a // math.ceil(A / 2)][a % math.ceil(A / 2)]
        ax.plot(range(H), means[:, a], color="black", lw=1.3)
        ax.fill_between(range(H),
                        means[:, a] - mean_std[:, a],
                        means[:, a] + mean_std[:, a],
                        color="#3b75af", alpha=0.25)
        ax.set_title(f"dim {a}", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("E0.3b — Action mean (line) ±1 σ (band) vs horizon", fontsize=11)
    fig.supxlabel("Horizon step h", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, out_dir / "plot_action_mean_horizon")


def _plot_e0_3c(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not rows:
        return
    steps = sorted({r["diffusion_steps"] for r in rows})

    def _pick(field: str) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {s: [] for s in steps}
        for r in rows:
            v = r.get(field)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out[r["diffusion_steps"]].append(float(v))
        return out

    lat = _pick("latency_s_mean")
    sigma = _pick("action_std_mean")
    err = _pick("action_l1_vs_gt")

    # latency
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    means = [np.mean(lat[s]) if lat[s] else float("nan") for s in steps]
    stds = [np.std(lat[s]) if len(lat[s]) > 1 else 0 for s in steps]
    ax.errorbar(steps, means, yerr=stds, fmt="o-", color="#3b75af", capsize=3)
    ax.set_xlabel("DiT inference steps")
    ax.set_ylabel("Latency (s)")
    ax.set_title("E0.3c — Latency vs diffusion steps")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_steps_vs_latency")

    # action seed-std
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    means = [np.mean(sigma[s]) if sigma[s] else float("nan") for s in steps]
    stds = [np.std(sigma[s]) if len(sigma[s]) > 1 else 0 for s in steps]
    ax.errorbar(steps, means, yerr=stds, fmt="o-", color="#c14b4b", capsize=3)
    ax.set_xlabel("DiT inference steps")
    ax.set_ylabel(r"mean $\sigma_{\mathrm{seed}}$")
    ax.set_title("E0.3c — Action variance vs diffusion steps")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_steps_vs_action_std")

    # action error
    if any(err[s] for s in steps):
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        means = [np.mean(err[s]) if err[s] else float("nan") for s in steps]
        stds = [np.std(err[s]) if len(err[s]) > 1 else 0 for s in steps]
        ax.errorbar(steps, means, yerr=stds, fmt="o-", color="#3b8b58", capsize=3)
        ax.set_xlabel("DiT inference steps")
        ax.set_ylabel("L1 vs ground truth")
        ax.set_title("E0.3c — Open-loop error vs diffusion steps")
        fig.tight_layout()
        _save_fig(fig, out_dir / "plot_steps_vs_action_error")


def _plot_e0_3d(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not rows:
        return
    shifts = sorted({r["sigma_shift"] for r in rows if r.get("sigma_shift") is not None})
    if not shifts:
        return
    sigma = {s: [r["action_std_mean"] for r in rows if r["sigma_shift"] == s and r.get("action_std_mean") is not None]
             for s in shifts}
    lat = {s: [r["latency_s_mean"] for r in rows if r["sigma_shift"] == s and r.get("latency_s_mean") is not None]
           for s in shifts}

    fig, ax1 = plt.subplots(figsize=(5.6, 3.2))
    ax2 = ax1.twinx()
    means_sigma = [np.mean(sigma[s]) if sigma[s] else float("nan") for s in shifts]
    means_lat = [np.mean(lat[s]) if lat[s] else float("nan") for s in shifts]
    ax1.plot(shifts, means_sigma, "o-", color="#c14b4b", label=r"$\sigma_{\mathrm{seed}}$")
    ax2.plot(shifts, means_lat, "s--", color="#3b75af", label="latency")
    ax1.set_xlabel(r"$\sigma_{\mathrm{shift}}$ (flow-matching schedule shift)")
    ax1.set_ylabel(r"mean $\sigma_{\mathrm{seed}}$", color="#c14b4b")
    ax2.set_ylabel("Latency (s)", color="#3b75af")
    ax1.tick_params(axis="y", colors="#c14b4b")
    ax2.tick_params(axis="y", colors="#3b75af")
    ax1.set_title("E0.3d — Sampling-parameter sweep")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_sigma_shift_vs_action_std")


def _plot_e0_3e(seed_arrays_per_example: list[np.ndarray], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    valid = [a for a in seed_arrays_per_example if a.size and a.shape[0] >= 2]
    if not valid:
        return
    H = min(a.shape[1] for a in valid)
    cropped = [a[:, :H, :] for a in valid]
    # std across seeds, then mean across dims, then mean across examples
    avg = np.stack([a.std(axis=0).mean(axis=-1) for a in cropped], axis=0)  # (E, H)
    mu = avg.mean(axis=0)
    sd = avg.std(axis=0) if avg.shape[0] > 1 else np.zeros_like(mu)

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(range(H), mu, color="black", lw=1.6, label=r"$\bar{\sigma}_{\mathrm{seed}}(h)$")
    ax.fill_between(range(H), mu - sd, mu + sd, color="#3b75af", alpha=0.25,
                    label="±1 σ across examples")
    # Highlight horizon thirds
    third = H // 3
    ax.axvspan(0, third, alpha=0.05, color="green")
    ax.axvspan(third, 2 * third, alpha=0.05, color="orange")
    ax.axvspan(2 * third, H, alpha=0.05, color="red")
    ax.set_xlabel("Horizon step h")
    ax.set_ylabel(r"mean $\sigma_{\mathrm{seed}}$ across dims")
    ax.set_title("E0.3e — Per-horizon stability (early vs late actions)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_horizon_stability")


# ============================================================================
# Sub-experiments
# ============================================================================


def run_e0_3a(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              num_runs: int, out_dir: Path) -> dict[str, Any]:
    """Same input, same seed → expected ≈ 0 across runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary_max_diffs: list[float] = []

    for ex_id, obs in examples:
        ref: np.ndarray | None = None
        per_run: list[dict[str, Any]] = []
        for r in range(num_runs):
            res = _run(policy, obs, seed=0, deterministic=True)
            if ref is None or ref.shape != res.action_chunk.shape:
                ref = res.action_chunk.copy()
                max_diff = 0.0
                mean_diff = 0.0
            else:
                max_diff = float(np.max(np.abs(res.action_chunk - ref))) if res.action_chunk.size else 0.0
                mean_diff = float(np.mean(np.abs(res.action_chunk - ref))) if res.action_chunk.size else 0.0
            row = {
                "example_id": ex_id,
                "run_index": r,
                "seed": 0,
                "latency_s": res.elapsed_s,
                "max_abs_diff_action": max_diff,
                "mean_abs_diff_action": mean_diff,
                "has_nan": res.has_nan,
            }
            rows.append(row)
            per_run.append(row)
            summary_max_diffs.append(max_diff)
        _plot_e0_3a(per_run, out_dir / ex_id)

    _plot_e0_3a(rows, out_dir)
    _enable_determinism(False)
    _write_csv(rows, out_dir / "per_example.csv")
    summary = {
        "num_runs_per_example": num_runs,
        "max_abs_diff_action": _stats(summary_max_diffs),
        "deterministic_pass": all(d <= 1e-4 for d in summary_max_diffs) if summary_max_diffs else False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_e0_3b(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              num_seeds: int, out_dir: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Same input, different seeds → noise floor σ_seed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seed_arrays: list[np.ndarray] = []  # per example: (S, H, A)

    for ex_id, obs in examples:
        chunks: list[np.ndarray] = []
        latencies: list[float] = []
        for s in range(num_seeds):
            res = _run(policy, obs, seed=s, deterministic=False)
            if not res.action_chunk.size:
                continue
            chunks.append(res.action_chunk)
            latencies.append(res.elapsed_s)
            rows.append({
                "example_id": ex_id,
                "seed": s,
                "latency_s": res.elapsed_s,
                "action_min": float(np.min(res.action_chunk)),
                "action_max": float(np.max(res.action_chunk)),
                "has_nan": res.has_nan,
            })
        if not chunks:
            continue
        # Align shapes (defensively)
        H = min(c.shape[0] for c in chunks)
        A = min(c.shape[1] for c in chunks)
        stack = np.stack([c[:H, :A] for c in chunks], axis=0)  # (S, H, A)
        seed_arrays.append(stack)

    _plot_e0_3b(seed_arrays, out_dir)
    _write_csv(rows, out_dir / "per_example.csv")

    # Noise-floor artifact
    summary = {"num_seeds_per_example": num_seeds, "examples": []}
    if seed_arrays:
        H = min(a.shape[1] for a in seed_arrays)
        A = min(a.shape[2] for a in seed_arrays)
        cropped = [a[:, :H, :A] for a in seed_arrays]
        per_h_a = np.stack([a.std(axis=0) for a in cropped], axis=0).mean(axis=0)  # (H, A)
        summary["sigma_seed_per_horizon_per_dim"] = per_h_a.tolist()
        summary["sigma_seed_per_horizon_mean"] = per_h_a.mean(axis=-1).tolist()
        summary["sigma_seed_overall_mean"] = float(per_h_a.mean())
        summary["sigma_seed_overall_p90"] = float(np.quantile(per_h_a, 0.90))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Noise-floor JSON used downstream as significance threshold
    noise_floor_path = out_dir.parent / "noise_floor.json"
    noise_floor_path.write_text(json.dumps({
        "definition": "sigma_seed[h, a] = std across seeds of action_chunk[h, a], averaged across examples.",
        "recommended_threshold_factor": 2.0,
        "sigma_seed_per_horizon_per_dim": summary.get("sigma_seed_per_horizon_per_dim"),
        "sigma_seed_overall_mean": summary.get("sigma_seed_overall_mean"),
        "sigma_seed_overall_p90": summary.get("sigma_seed_overall_p90"),
    }, indent=2))

    return summary, seed_arrays


def run_e0_3c(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              steps_list: list[int], num_seeds: int, gt_chunks: dict[str, np.ndarray],
              out_dir: Path) -> dict[str, Any]:
    """Diffusion-step ablation (1 / 2 / 4 / default DiT steps)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head
    orig_n, orig_mask = _set_diffusion_steps(head, int(getattr(head, "num_inference_steps", 16)))
    _restore_diffusion_steps(head, orig_n, orig_mask)  # no-op; just to capture origins

    rows: list[dict[str, Any]] = []
    try:
        for steps in steps_list:
            on, om = _set_diffusion_steps(head, steps)
            try:
                for ex_id, obs in examples:
                    chunks: list[np.ndarray] = []
                    latencies: list[float] = []
                    for s in range(num_seeds):
                        res = _run(policy, obs, seed=s)
                        if res.action_chunk.size:
                            chunks.append(res.action_chunk)
                            latencies.append(res.elapsed_s)
                    if not chunks:
                        continue
                    H = min(c.shape[0] for c in chunks)
                    A = min(c.shape[1] for c in chunks)
                    stack = np.stack([c[:H, :A] for c in chunks], axis=0)
                    sigma_mean = float(stack.std(axis=0).mean())
                    gt = gt_chunks.get(ex_id)
                    l1 = None
                    if gt is not None and gt.size:
                        Hg = min(gt.shape[0], H)
                        Ag = min(gt.shape[1], A)
                        l1 = float(np.mean(np.abs(stack.mean(axis=0)[:Hg, :Ag] - gt[:Hg, :Ag])))
                    rows.append({
                        "example_id": ex_id,
                        "diffusion_steps": int(steps),
                        "num_seeds": num_seeds,
                        "latency_s_mean": float(np.mean(latencies)),
                        "latency_s_std": float(np.std(latencies)) if len(latencies) > 1 else 0.0,
                        "action_std_mean": sigma_mean,
                        "action_l1_vs_gt": l1,
                    })
            finally:
                _restore_diffusion_steps(head, on, om)
    finally:
        _restore_diffusion_steps(head, orig_n, orig_mask)

    _plot_e0_3c(rows, out_dir)
    _write_csv(rows, out_dir / "per_example.csv")

    # Aggregate per step
    by_step: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        d = by_step.setdefault(int(r["diffusion_steps"]), {"latency": [], "sigma": [], "l1": []})
        d["latency"].append(r["latency_s_mean"])
        d["sigma"].append(r["action_std_mean"])
        if r["action_l1_vs_gt"] is not None:
            d["l1"].append(r["action_l1_vs_gt"])
    summary = {
        "steps_evaluated": sorted(by_step.keys()),
        "per_step": {
            int(k): {
                "latency_s": _stats(v["latency"]),
                "sigma_seed": _stats(v["sigma"]),
                "l1_vs_gt": _stats(v["l1"]) if v["l1"] else None,
            } for k, v in by_step.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_e0_3d(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              sigma_shifts: list[float], num_seeds: int, out_dir: Path) -> dict[str, Any]:
    """Sigma-shift sweep (best-effort; depends on action head exposing it)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head
    if not hasattr(head, "sigma_shift"):
        msg = "action_head has no `sigma_shift` attribute; E0.3d skipped."
        logger.warning(msg)
        (out_dir / "summary.json").write_text(json.dumps({"skipped": True, "reason": msg}, indent=2))
        return {"skipped": True, "reason": msg}

    orig = float(head.sigma_shift)
    rows: list[dict[str, Any]] = []
    try:
        for shift in sigma_shifts:
            head.sigma_shift = float(shift)
            for ex_id, obs in examples:
                chunks: list[np.ndarray] = []
                latencies: list[float] = []
                for s in range(num_seeds):
                    res = _run(policy, obs, seed=s)
                    if res.action_chunk.size:
                        chunks.append(res.action_chunk)
                        latencies.append(res.elapsed_s)
                if not chunks:
                    continue
                H = min(c.shape[0] for c in chunks)
                A = min(c.shape[1] for c in chunks)
                stack = np.stack([c[:H, :A] for c in chunks], axis=0)
                sigma_mean = float(stack.std(axis=0).mean())
                rows.append({
                    "example_id": ex_id,
                    "sigma_shift": float(shift),
                    "num_seeds": num_seeds,
                    "latency_s_mean": float(np.mean(latencies)),
                    "action_std_mean": sigma_mean,
                })
    finally:
        head.sigma_shift = orig

    _plot_e0_3d(rows, out_dir)
    _write_csv(rows, out_dir / "per_example.csv")

    by_shift: dict[float, dict[str, list[float]]] = {}
    for r in rows:
        d = by_shift.setdefault(float(r["sigma_shift"]), {"latency": [], "sigma": []})
        d["latency"].append(r["latency_s_mean"])
        d["sigma"].append(r["action_std_mean"])
    summary = {
        "sigma_shifts_evaluated": sorted(by_shift.keys()),
        "per_shift": {
            float(k): {
                "latency_s": _stats(v["latency"]),
                "sigma_seed": _stats(v["sigma"]),
            } for k, v in by_shift.items()
        },
        "original_sigma_shift": orig,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_e0_3e(seed_arrays: list[np.ndarray], out_dir: Path) -> dict[str, Any]:
    """Per-horizon stability — re-uses the per-seed arrays from E0.3b."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not seed_arrays:
        (out_dir / "summary.json").write_text(json.dumps({"skipped": True}, indent=2))
        return {"skipped": True}

    H = min(a.shape[1] for a in seed_arrays)
    cropped = [a[:, :H, :] for a in seed_arrays]
    avg = np.stack([a.std(axis=0).mean(axis=-1) for a in cropped], axis=0)  # (E, H)
    mu = avg.mean(axis=0)

    third = max(1, H // 3)
    early = float(mu[:third].mean())
    middle = float(mu[third:2 * third].mean())
    late = float(mu[2 * third:].mean())

    summary = {
        "horizon_length": int(H),
        "sigma_seed_per_horizon_mean": mu.tolist(),
        "early_third_mean": early,
        "middle_third_mean": middle,
        "late_third_mean": late,
        "late_to_early_ratio": (late / early) if early > 0 else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-example CSV
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(cropped):
        per_h = a.std(axis=0).mean(axis=-1)
        for h in range(H):
            rows.append({"example_index": i, "horizon": h, "sigma_seed_mean": float(per_h[h])})
    _write_csv(rows, out_dir / "per_example.csv")
    _plot_e0_3e(seed_arrays, out_dir)
    return summary


# ============================================================================
# Helpers
# ============================================================================


def _stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        out.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _select_examples(manifest: list[dict[str, Any]], num: int) -> list[dict[str, Any]]:
    """Pick `num` examples spread across distinct task groups when possible."""
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


def _build_combined_report(out_dir: Path,
                           ex_ids: list[str],
                           a_summary: dict[str, Any],
                           b_summary: dict[str, Any],
                           c_summary: dict[str, Any],
                           d_summary: dict[str, Any],
                           e_summary: dict[str, Any]) -> None:
    lines = ["# Stage 0 / E0.3 — Action Stability Report",
             "",
             f"- Examples evaluated: {len(ex_ids)} ({', '.join(ex_ids)})",
             "",
             "## Headline numbers",
             "",
             f"- **σ_seed** (overall mean across (h, a)): "
             f"`{b_summary.get('sigma_seed_overall_mean'):.6f}`"
             if b_summary.get("sigma_seed_overall_mean") is not None else "- σ_seed: n/a",
             f"- **σ_seed** p90: "
             f"`{b_summary.get('sigma_seed_overall_p90'):.6f}`"
             if b_summary.get("sigma_seed_overall_p90") is not None else "",
             f"- **Determinism check (E0.3a)**: "
             f"max-abs-diff p90 = `{a_summary.get('max_abs_diff_action', {}).get('p90', float('nan')):.2e}` "
             f"({'PASS' if a_summary.get('deterministic_pass') else 'FAIL'})",
             "",
             "## E0.3a — Same input, same seed",
             "",
             f"- N runs / example: {a_summary.get('num_runs_per_example')}",
             f"- See `e0_3a_determinism/plot_max_abs_diff.png`.",
             "",
             "## E0.3b — Same input, different seeds (noise floor)",
             "",
             f"- N seeds / example: {b_summary.get('num_seeds_per_example')}",
             "- Plots:",
             "  - `e0_3b_seed_variance/plot_std_heatmap.png` (horizon × dim heatmap)",
             "  - `e0_3b_seed_variance/plot_std_per_horizon.png`",
             "  - `e0_3b_seed_variance/plot_std_per_dim_box.png`",
             "  - `e0_3b_seed_variance/plot_action_mean_horizon.png`",
             "- Noise floor saved to `noise_floor.json`. Use `2 × σ_seed` as the perturbation-significance threshold.",
             "",
             "## E0.3c — Diffusion-step ablation",
             "",
             f"- Steps evaluated: {c_summary.get('steps_evaluated')}",
             "- Plots: `e0_3c_diffusion_steps/plot_steps_vs_*.png`",
             "",
             "## E0.3d — Sampling-parameter sweep (sigma_shift)",
             "",
             f"- Skipped: {bool(d_summary.get('skipped', False))}",
             "- Plot (if not skipped): `e0_3d_noise_schedule/plot_sigma_shift_vs_action_std.png`",
             "",
             "## E0.3e — Per-horizon stability",
             "",
             f"- Horizon length: {e_summary.get('horizon_length')}",
             f"- Early third σ: `{e_summary.get('early_third_mean'):.6f}`"
             if e_summary.get("early_third_mean") is not None else "",
             f"- Middle third σ: `{e_summary.get('middle_third_mean'):.6f}`"
             if e_summary.get("middle_third_mean") is not None else "",
             f"- Late third σ: `{e_summary.get('late_third_mean'):.6f}`"
             if e_summary.get("late_third_mean") is not None else "",
             f"- Late-to-early ratio: `{e_summary.get('late_to_early_ratio')}`",
             "- Plot: `e0_3e_horizon/plot_horizon_stability.png`",
             "",
             "## Recommended thresholds for downstream stages",
             "",
             "Δa_i is **significant** iff:",
             "",
             "    Δa_i > 2 × σ_seed[h, a]        (per-horizon, per-dim)",
             "    or",
             "    Δa_i > 2 × σ_seed_overall_p90  (single conservative threshold)",
             ""]
    (out_dir / "combined_report.md").write_text("\n".join(lines))


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_examples", type=int, default=3,
                   help="Number of examples (spread across task groups) to use.")
    p.add_argument("--num_seeds", type=int, default=16,
                   help="Seed sweep width for E0.3b/E0.3e.")
    p.add_argument("--num_determinism_runs", type=int, default=8,
                   help="Repeat count for E0.3a (same seed).")
    p.add_argument("--num_seeds_ablation", type=int, default=4,
                   help="Seed count per setting in E0.3c/E0.3d.")
    p.add_argument("--diffusion_steps", default="1,2,4,8",
                   help="Comma-separated DiT step counts for E0.3c.")
    p.add_argument("--sigma_shifts", default="3.0,5.0,7.0",
                   help="Comma-separated sigma_shift values for E0.3d.")
    p.add_argument("--skip_e0_3a", action="store_true")
    p.add_argument("--skip_e0_3c", action="store_true")
    p.add_argument("--skip_e0_3d", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_neurips_matplotlib()

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

    examples: list[tuple[str, dict[str, Any]]] = []
    gt_chunks: dict[str, np.ndarray] = {}
    for entry in chosen:
        try:
            obs = build_obs(entry)
        except Exception as e:
            logger.warning("Skipping %s: %s", entry.get("example_id"), e)
            continue
        ex_id = entry["example_id"]
        examples.append((ex_id, obs))
        gt = np.asarray(entry.get("gt_action_chunk", []), dtype=np.float64)
        if gt.size:
            gt_chunks[ex_id] = gt
    if not examples:
        logger.error("No usable examples after building observations.")
        sys.exit(1)

    ex_ids = [eid for eid, _ in examples]
    logger.info("Running stability over %d examples: %s", len(ex_ids), ex_ids)

    a_summary: dict[str, Any] = {"num_runs_per_example": 0, "deterministic_pass": False,
                                 "max_abs_diff_action": {}}
    if not args.skip_e0_3a:
        logger.info("E0.3a — same-input/same-seed determinism check ...")
        a_summary = run_e0_3a(policy, examples, args.num_determinism_runs,
                              out_dir / "e0_3a_determinism")

    logger.info("E0.3b — seed sweep (noise floor) ...")
    b_summary, seed_arrays = run_e0_3b(policy, examples, args.num_seeds,
                                       out_dir / "e0_3b_seed_variance")

    c_summary: dict[str, Any] = {}
    if not args.skip_e0_3c:
        steps_list = sorted({int(x) for x in args.diffusion_steps.split(",") if x.strip()})
        logger.info("E0.3c — diffusion-step ablation: %s", steps_list)
        c_summary = run_e0_3c(policy, examples, steps_list,
                              args.num_seeds_ablation, gt_chunks,
                              out_dir / "e0_3c_diffusion_steps")

    d_summary: dict[str, Any] = {}
    if not args.skip_e0_3d:
        shifts = sorted({float(x) for x in args.sigma_shifts.split(",") if x.strip()})
        logger.info("E0.3d — sigma_shift sweep: %s", shifts)
        d_summary = run_e0_3d(policy, examples, shifts,
                              args.num_seeds_ablation, out_dir / "e0_3d_noise_schedule")

    logger.info("E0.3e — per-horizon stability ...")
    e_summary = run_e0_3e(seed_arrays, out_dir / "e0_3e_horizon")

    _build_combined_report(out_dir, ex_ids, a_summary, b_summary, c_summary, d_summary, e_summary)
    (out_dir / "combined_report.json").write_text(json.dumps({
        "examples": ex_ids,
        "e0_3a": a_summary,
        "e0_3b": b_summary,
        "e0_3c": c_summary,
        "e0_3d": d_summary,
        "e0_3e": e_summary,
    }, indent=2))

    logger.info("Done. Combined report -> %s", out_dir / "combined_report.md")


if __name__ == "__main__":
    main()
