#!/usr/bin/env python3
"""Stage 0 / E0.8 — Profiling and Compute Measurement for DreamZero-DROID.

A complete efficiency test suite. Implements all five Stage-0.8 sub-experiments
and emits paper-ready artifacts that downstream stages (perturbation, allocation)
use to design block sizes and batching strategies.

    E0.8a — End-to-end latency breakdown (preprocessing, text/image/VAE encoder,
            DiT, VAE decoder, action extraction). CUDA-event accurate.
    E0.8b — GPU memory: parameters, peak, reserved, KV cache, cross-attention
            cache, activation memory.
    E0.8c — Batch-size scaling (1, 2, 4, 8): latency, throughput, memory,
            scaling efficiency, projected speedup for batched perturbation.
    E0.8d — Scaling with diffusion steps and image resolution. Reports latency,
            memory, action variance.
    E0.8e — Perturbation-cost projection. Given a candidate block grid, project
            sequential vs batched forward-pass cost using E0.8a/c numbers.

Per-experiment artifacts (CSV + JSON + PNG/PDF figures) under <output_dir>/<exp>.
Top-level: combined_report.md, combined_report.json, plus a paper-ready
summary table baseline_efficiency.csv.

Example:

    python scripts/stage0/efficiency_profile.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_examples 3 \\
        --num_warmup 2 \\
        --num_runs 8 \\
        --batch_sizes 1,2,4,8 \\
        --diffusion_steps 1,2,4,8 \\
        --resolutions 180x320,128x224 \\
        --block_sizes "1x1x1,1x2x2,1x4x4,1x8x8,1x16x16" \\
        --output_dir runs/stage0_efficiency
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
from typing import Any, Iterable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage0.efficiency")

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
    EMBODIMENT_TAG,
    build_obs,
    configure_neurips_matplotlib,
    flatten_action_dict,
    init_dist_single_process,
    reset_causal_state,
)


# ============================================================================
# CUDA-event timing hooks
# ============================================================================


@dataclass
class StageTimer:
    """Accumulates GPU-wall time across multiple calls of a hooked module."""
    name: str
    events: list[tuple[Any, Any]] = field(default_factory=list)
    _pending_start: Any = None

    def hook_pre(self, _mod, _inputs):
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        self._pending_start = s

    def hook_post(self, _mod, _inputs, _outputs):
        if self._pending_start is None:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append((self._pending_start, e))
        self._pending_start = None

    def elapsed_ms(self) -> float:
        if not self.events:
            return 0.0
        return float(sum(s.elapsed_time(e) for s, e in self.events))

    def call_count(self) -> int:
        return len(self.events)

    def reset(self) -> None:
        self.events = []
        self._pending_start = None


def _resolve(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _install_timing_hooks(head: Any) -> tuple[dict[str, StageTimer], list[Any]]:
    """Return per-stage timers and the hook handles to remove later."""
    candidates: list[tuple[str, list[str]]] = [
        ("text_encoder", ["text_encoder"]),
        ("image_encoder", ["image_encoder"]),
        ("vae_encoder", ["vae.encoder", "vae.model.encoder"]),
        ("vae_decoder", ["vae.decoder", "vae.model.decoder"]),
        ("dit_patch_embed", ["model.patch_embedding"]),
        ("dit_full", ["model"]),
        ("dit_head", ["model.head"]),
    ]
    timers: dict[str, StageTimer] = {}
    handles: list[Any] = []
    for name, paths in candidates:
        for p in paths:
            mod = _resolve(head, p)
            if mod is None or not hasattr(mod, "register_forward_hook"):
                continue
            timer = StageTimer(name=name)
            timers[name] = timer
            try:
                handles.append(mod.register_forward_pre_hook(timer.hook_pre))
                handles.append(mod.register_forward_hook(timer.hook_post))
            except Exception as e:
                logger.warning("Could not hook %s (%s): %s", name, p, e)
            break  # first resolved path wins
    return timers, handles


def _remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _collect_timer_breakdown(timers: dict[str, StageTimer], total_wall_s: float) -> dict[str, float]:
    """Collect per-stage ms (and call counts) plus a residual 'other' bucket."""
    out: dict[str, float] = {}
    accounted_ms = 0.0
    for name, timer in timers.items():
        ms = timer.elapsed_ms()
        out[name + "_ms"] = ms
        out[name + "_calls"] = timer.call_count()
        if name in ("dit_full",):
            # Use dit_full as the DiT total; do not double-count patch/head.
            accounted_ms += ms
        elif name in ("dit_patch_embed", "dit_head"):
            # Sub-stages of dit_full; not added to accounted_ms.
            pass
        else:
            accounted_ms += ms
    out["total_ms"] = total_wall_s * 1000.0
    out["other_ms"] = max(0.0, out["total_ms"] - accounted_ms)
    out["accounted_fraction"] = (accounted_ms / out["total_ms"]) if out["total_ms"] > 0 else 0.0
    return out


# ============================================================================
# Memory measurement
# ============================================================================


def _module_param_bytes(module: Any) -> int:
    try:
        return sum(p.numel() * p.element_size() for p in module.parameters())
    except Exception:
        return 0


def _kv_cache_bytes(head: Any) -> dict[str, int]:
    out: dict[str, int] = {"kv_cache1": 0, "kv_cache_neg": 0,
                           "crossattn_cache": 0, "crossattn_cache_neg": 0}
    for k in list(out.keys()):
        cache = getattr(head, k, None)
        if cache is None:
            continue
        if isinstance(cache, list):
            out[k] = int(sum(t.numel() * t.element_size() for t in cache if torch.is_tensor(t)))
        elif torch.is_tensor(cache):
            out[k] = int(cache.numel() * cache.element_size())
    out["kv_cache_total"] = int(sum(out.values()))
    return out


@dataclass
class MemoryReport:
    parameters_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    kv_cache_bytes: dict[str, int]
    activation_bytes_estimate: int        # peak_alloc - parameters - kv_cache
    after_inference_alloc_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters_bytes": self.parameters_bytes,
            "parameters_mb": self.parameters_bytes / 1024 / 1024,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_allocated_mb": self.peak_allocated_bytes / 1024 / 1024,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "peak_reserved_mb": self.peak_reserved_bytes / 1024 / 1024,
            "kv_cache_bytes": self.kv_cache_bytes,
            "kv_cache_total_mb": self.kv_cache_bytes.get("kv_cache_total", 0) / 1024 / 1024,
            "activation_bytes_estimate": self.activation_bytes_estimate,
            "activation_mb_estimate": self.activation_bytes_estimate / 1024 / 1024,
            "after_inference_alloc_bytes": self.after_inference_alloc_bytes,
            "after_inference_alloc_mb": self.after_inference_alloc_bytes / 1024 / 1024,
        }


def _capture_memory(policy: GrootSimPolicy) -> MemoryReport:
    head = policy.trained_model.action_head if hasattr(policy.trained_model, "action_head") else None
    params = _module_param_bytes(policy.trained_model)
    if not torch.cuda.is_available():
        return MemoryReport(
            parameters_bytes=params, peak_allocated_bytes=0, peak_reserved_bytes=0,
            kv_cache_bytes={"kv_cache_total": 0}, activation_bytes_estimate=0,
            after_inference_alloc_bytes=0,
        )
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    kv = _kv_cache_bytes(head) if head is not None else {"kv_cache_total": 0}
    activation_est = max(0, peak_alloc - params - int(kv.get("kv_cache_total", 0)))
    after_alloc = int(torch.cuda.memory_allocated())
    return MemoryReport(
        parameters_bytes=params,
        peak_allocated_bytes=peak_alloc,
        peak_reserved_bytes=peak_reserved,
        kv_cache_bytes=kv,
        activation_bytes_estimate=activation_est,
        after_inference_alloc_bytes=after_alloc,
    )


# ============================================================================
# Inference primitives
# ============================================================================


def _replicate_obs(obs: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Replicate a single-example obs (from build_obs) to batch_size B.

    build_obs gives: video.* (1,H,W,3) uint8 ; state.* (1,D) float ; annotation.* str.
    After replication: video.* (B,1,H,W,3) ; state.* (B,1,D) ; annotation.* (B,) str array.
    """
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            v_t = np.expand_dims(v, axis=0)               # (1, *original)
            out[k] = np.repeat(v_t, batch_size, axis=0)   # (B, *original)
        elif isinstance(v, str):
            out[k] = np.array([v] * batch_size)
        else:
            out[k] = v
    return out


@contextmanager
def _override_diffusion_steps(head: Any, n: int | None):
    if n is None:
        yield
        return
    orig_n = int(getattr(head, "num_inference_steps", 16))
    orig_mask = list(getattr(head, "dit_step_mask", [True] * orig_n))
    head.num_inference_steps = int(n)
    head.dit_step_mask = [True] * int(n)
    try:
        yield
    finally:
        head.num_inference_steps = orig_n
        head.dit_step_mask = orig_mask


@contextmanager
def _override_resolution(head: Any, hw: tuple[int, int] | None):
    if hw is None:
        yield
        return
    cfg = getattr(head, "config", None)
    if cfg is None:
        yield
        return
    orig_h = getattr(cfg, "target_video_height", None)
    orig_w = getattr(cfg, "target_video_width", None)
    cfg.target_video_height = int(hw[0])
    cfg.target_video_width = int(hw[1])
    try:
        yield
    finally:
        cfg.target_video_height = orig_h
        cfg.target_video_width = orig_w


def _run_one(policy: GrootSimPolicy, obs: dict[str, Any],
             timers: dict[str, StageTimer] | None = None,
             seed: int = 0) -> tuple[float, np.ndarray, dict[str, float], MemoryReport]:
    """Single inference call with optional per-stage timing and memory capture."""
    if timers is not None:
        for t in timers.values():
            t.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
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
    breakdown = _collect_timer_breakdown(timers, elapsed) if timers else {"total_ms": elapsed * 1000.0}
    mem = _capture_memory(policy)
    return elapsed, chunk, breakdown, mem


# ============================================================================
# Sub-experiments
# ============================================================================


def run_e0_8a(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              num_warmup: int, num_runs: int, out_dir: Path) -> dict[str, Any]:
    """End-to-end latency breakdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head
    timers, handles = _install_timing_hooks(head)
    rows: list[dict[str, Any]] = []
    try:
        # Warmup (lazy compile, kernel load, etc.)
        for ex_id, obs in examples[: max(1, len(examples))]:
            for _ in range(num_warmup):
                _run_one(policy, obs, timers=timers, seed=0)
            for r in range(num_runs):
                elapsed, chunk, breakdown, mem = _run_one(policy, obs, timers=timers, seed=r)
                row = {"example_id": ex_id, "run_index": r, **breakdown,
                       "peak_alloc_mb": mem.peak_allocated_bytes / 1024 / 1024}
                rows.append(row)
    finally:
        _remove_hooks(handles)

    _write_csv(rows, out_dir / "per_call.csv")

    if not rows:
        (out_dir / "summary.json").write_text("{}")
        return {}

    keys = [k for k in rows[0].keys() if k.endswith("_ms")]
    summary = {}
    for k in keys:
        summary[k] = _stats([r[k] for r in rows if r.get(k) is not None])
    summary["accounted_fraction_mean"] = float(np.mean([r.get("accounted_fraction", 0.0) for r in rows]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_e0_8a(rows, summary, out_dir)
    return summary


def run_e0_8b(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              num_runs: int, out_dir: Path) -> dict[str, Any]:
    """GPU memory measurement at batch_size = 1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head
    rows: list[dict[str, Any]] = []
    for ex_id, obs in examples:
        for r in range(num_runs):
            elapsed, _chunk, _bd, mem = _run_one(policy, obs, timers=None, seed=r)
            row = {"example_id": ex_id, "run_index": r,
                   **mem.to_dict(), "latency_s": elapsed}
            rows.append(row)
    _write_csv(rows, out_dir / "per_run.csv")

    if not rows:
        (out_dir / "summary.json").write_text("{}")
        return {}

    summary = {
        "parameters_mb": float(np.mean([r["parameters_mb"] for r in rows])),
        "peak_allocated_mb": _stats([r["peak_allocated_mb"] for r in rows]),
        "peak_reserved_mb": _stats([r["peak_reserved_mb"] for r in rows]),
        "kv_cache_total_mb": _stats([r["kv_cache_total_mb"] for r in rows]),
        "activation_mb_estimate": _stats([r["activation_mb_estimate"] for r in rows]),
        "after_inference_alloc_mb": _stats([r["after_inference_alloc_mb"] for r in rows]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_e0_8b(summary, out_dir)
    return summary


def run_e0_8c(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              batch_sizes: list[int], num_warmup: int, num_runs: int,
              out_dir: Path) -> dict[str, Any]:
    """Batch-size scaling. Picks the first viable example to keep memory bounded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not examples:
        (out_dir / "summary.json").write_text("{}")
        return {}
    ex_id, obs = examples[0]

    rows: list[dict[str, Any]] = []
    notes: dict[int, str] = {}
    for B in batch_sizes:
        try:
            obs_b = _replicate_obs(obs, B)
            # warmup
            for _ in range(num_warmup):
                _run_one(policy, obs_b, timers=None, seed=0)
            run_lats: list[float] = []
            run_peaks: list[float] = []
            for r in range(num_runs):
                elapsed, chunk, _bd, mem = _run_one(policy, obs_b, timers=None, seed=r)
                run_lats.append(elapsed)
                run_peaks.append(mem.peak_allocated_bytes / 1024 / 1024)
                rows.append({
                    "example_id": ex_id, "batch_size": B, "run_index": r,
                    "latency_s": elapsed,
                    "throughput_samples_per_s": (B / elapsed) if elapsed > 0 else 0.0,
                    "peak_alloc_mb": mem.peak_allocated_bytes / 1024 / 1024,
                    "peak_reserved_mb": mem.peak_reserved_bytes / 1024 / 1024,
                })
        except torch.cuda.OutOfMemoryError as e:
            notes[B] = f"OOM at batch_size={B}: {e}"
            logger.warning("Batch=%d OOM — stopping batch scaling sweep.", B)
            torch.cuda.empty_cache()
            break
        except Exception as e:
            notes[B] = f"Failed at batch_size={B}: {e}"
            logger.warning("Batch=%d failed: %s", B, e)
            continue
    _write_csv(rows, out_dir / "per_batch.csv")

    by_b: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        d = by_b.setdefault(int(r["batch_size"]), {"lat": [], "thr": [], "mem": []})
        d["lat"].append(r["latency_s"])
        d["thr"].append(r["throughput_samples_per_s"])
        d["mem"].append(r["peak_alloc_mb"])
    summary: dict[str, Any] = {
        "batch_sizes_evaluated": sorted(by_b.keys()),
        "per_batch": {
            int(B): {
                "latency_s": _stats(d["lat"]),
                "throughput_samples_per_s": _stats(d["thr"]),
                "peak_alloc_mb": _stats(d["mem"]),
            } for B, d in by_b.items()
        },
        "notes": notes,
    }
    # Scaling efficiency: throughput(B) / (B * throughput(1))
    if 1 in by_b and by_b[1]["thr"]:
        thr1 = float(np.mean(by_b[1]["thr"]))
        eff = {}
        for B, d in by_b.items():
            thrB = float(np.mean(d["thr"])) if d["thr"] else 0.0
            eff[int(B)] = (thrB / (B * thr1)) if thr1 > 0 else 0.0
        summary["scaling_efficiency"] = eff
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_e0_8c(summary, out_dir)
    return summary


def run_e0_8d(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              diffusion_steps: list[int], resolutions: list[tuple[int, int]],
              num_seeds: int, out_dir: Path) -> dict[str, Any]:
    """Diffusion-step + image-resolution sweep."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head

    rows: list[dict[str, Any]] = []
    if not examples:
        (out_dir / "summary.json").write_text("{}")
        return {}
    ex_id, obs = examples[0]

    # Diffusion-step sweep at native resolution
    for steps in diffusion_steps:
        with _override_diffusion_steps(head, steps):
            chunks: list[np.ndarray] = []
            lats: list[float] = []
            peak_mb: list[float] = []
            for s in range(num_seeds):
                try:
                    elapsed, chunk, _bd, mem = _run_one(policy, obs, timers=None, seed=s)
                except Exception as e:
                    logger.warning("E0.8d steps=%d failed: %s", steps, e)
                    break
                lats.append(elapsed)
                peak_mb.append(mem.peak_allocated_bytes / 1024 / 1024)
                if chunk.size:
                    chunks.append(chunk)
            sigma = 0.0
            if len(chunks) >= 2:
                H = min(c.shape[0] for c in chunks)
                A = min(c.shape[1] for c in chunks)
                sigma = float(np.stack([c[:H, :A] for c in chunks]).std(axis=0).mean())
            rows.append({
                "axis": "diffusion_steps", "value": steps,
                "latency_s_mean": float(np.mean(lats)) if lats else float("nan"),
                "peak_alloc_mb_mean": float(np.mean(peak_mb)) if peak_mb else float("nan"),
                "action_std_mean": sigma,
                "num_seeds": len(lats),
            })

    # Resolution sweep at default diffusion steps
    for hw in resolutions:
        with _override_resolution(head, hw):
            chunks = []
            lats = []
            peak_mb = []
            for s in range(num_seeds):
                try:
                    elapsed, chunk, _bd, mem = _run_one(policy, obs, timers=None, seed=s)
                except Exception as e:
                    logger.warning("E0.8d res=%dx%d failed: %s", hw[0], hw[1], e)
                    break
                lats.append(elapsed)
                peak_mb.append(mem.peak_allocated_bytes / 1024 / 1024)
                if chunk.size:
                    chunks.append(chunk)
            sigma = 0.0
            if len(chunks) >= 2:
                H = min(c.shape[0] for c in chunks)
                A = min(c.shape[1] for c in chunks)
                sigma = float(np.stack([c[:H, :A] for c in chunks]).std(axis=0).mean())
            rows.append({
                "axis": "resolution", "value": f"{hw[0]}x{hw[1]}",
                "latency_s_mean": float(np.mean(lats)) if lats else float("nan"),
                "peak_alloc_mb_mean": float(np.mean(peak_mb)) if peak_mb else float("nan"),
                "action_std_mean": sigma,
                "num_seeds": len(lats),
            })

    _write_csv(rows, out_dir / "per_config.csv")
    summary = {"rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_e0_8d(rows, out_dir)
    return summary


def run_e0_8e(policy: GrootSimPolicy, examples: list[tuple[str, dict[str, Any]]],
              block_sizes: list[tuple[int, int, int]],
              latency_breakdown: dict[str, Any] | None,
              batch_summary: dict[str, Any] | None,
              num_seeds_for_significance: int,
              out_dir: Path) -> dict[str, Any]:
    """Project the cost of perturbation experiments using E0.8a/c numbers.

    For each candidate block size (Bt, Bh, Bw) over the (T_lat, H_lat, W_lat)
    DiT video latent, we compute:
        num_blocks = ceil(T_lat / Bt) * ceil(H_lat / Bh) * ceil(W_lat / Bw)
        sequential_cost  = num_blocks * latency_at_batch1
        batched_cost(B*) = ceil(num_blocks / B*) * latency_at_batchB*
                          for each measured B*
        speedup = sequential_cost / min(batched_cost over B*)
        total_with_seeds = best_cost * num_seeds_for_significance
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not examples:
        (out_dir / "summary.json").write_text("{}")
        return {}
    ex_id, obs = examples[0]

    # Probe latent shape with one forward pass (no hooks, no perturbation).
    _, _chunk, _, _mem = _run_one(policy, obs, timers=None, seed=0)
    head = policy.trained_model.action_head
    # Capture video_pred shape from a fresh forward pass.
    reset_causal_state(policy)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    with torch.inference_mode():
        _result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if not torch.is_tensor(video_pred):
        (out_dir / "summary.json").write_text(json.dumps({"skipped": True,
                                                          "reason": "video_pred not a tensor"}, indent=2))
        return {"skipped": True}

    shape = list(video_pred.shape)
    # Expect (B, T_lat, C, H_lat, W_lat); be tolerant about ordering.
    if len(shape) < 4:
        (out_dir / "summary.json").write_text(json.dumps({"skipped": True,
                                                          "reason": f"unexpected video_pred shape {shape}"},
                                                         indent=2))
        return {"skipped": True}
    if len(shape) == 5:
        # (B, T, C, H, W) — confirm: T should be smaller than C usually false; pick by position.
        _, T_lat, C_lat, H_lat, W_lat = shape
    elif len(shape) == 4:
        T_lat, C_lat, H_lat, W_lat = shape
    else:
        T_lat = shape[-3]; C_lat = shape[-3]; H_lat = shape[-2]; W_lat = shape[-1]

    # Latency reference numbers
    base_latency_s = 0.0
    if latency_breakdown:
        base_latency_s = float(latency_breakdown.get("total_ms", {}).get("median", 0.0)) / 1000.0
    if base_latency_s <= 0:
        elapsed, _c, _bd, _m = _run_one(policy, obs, timers=None, seed=0)
        base_latency_s = elapsed

    batch_latencies: dict[int, float] = {1: base_latency_s}
    if batch_summary and batch_summary.get("per_batch"):
        for Bk, stats in batch_summary["per_batch"].items():
            try:
                batch_latencies[int(Bk)] = float(stats["latency_s"]["median"])
            except Exception:
                continue

    rows: list[dict[str, Any]] = []
    for (bt, bh, bw) in block_sizes:
        bt = max(1, min(bt, T_lat))
        bh = max(1, min(bh, H_lat))
        bw = max(1, min(bw, W_lat))
        n_blocks = math.ceil(T_lat / bt) * math.ceil(H_lat / bh) * math.ceil(W_lat / bw)

        sequential_s = n_blocks * base_latency_s
        # Batched candidates
        best_B = 1
        best_s = sequential_s
        per_batch_cost: dict[int, float] = {}
        for Bk, lat_s in batch_latencies.items():
            cost = math.ceil(n_blocks / Bk) * lat_s
            per_batch_cost[Bk] = cost
            if cost < best_s:
                best_s = cost; best_B = Bk
        speedup = (sequential_s / best_s) if best_s > 0 else 0.0
        total_with_seeds = best_s * max(1, num_seeds_for_significance)

        rows.append({
            "block_size": f"{bt}x{bh}x{bw}",
            "Bt": bt, "Bh": bh, "Bw": bw,
            "num_blocks": n_blocks,
            "latent_T": T_lat, "latent_H": H_lat, "latent_W": W_lat,
            "sequential_total_s": sequential_s,
            **{f"batched_total_s_B{Bk}": v for Bk, v in per_batch_cost.items()},
            "best_batch": best_B,
            "best_total_s": best_s,
            "speedup_vs_sequential": speedup,
            "total_s_with_seed_repeats": total_with_seeds,
            "total_min_with_seed_repeats": total_with_seeds / 60.0,
        })

    _write_csv(rows, out_dir / "per_block_size.csv")
    summary = {
        "latent_shape": {"T": T_lat, "C": C_lat, "H": H_lat, "W": W_lat},
        "base_latency_s_at_batch1": base_latency_s,
        "batch_latencies_used_s": batch_latencies,
        "num_seeds_for_significance": num_seeds_for_significance,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_e0_8e(rows, out_dir, batch_latencies)
    return summary


# ============================================================================
# Plotting
# ============================================================================


def _save_fig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(str(base) + ext)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_e0_8a(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not rows:
        return

    stage_keys = [k for k in summary.keys()
                  if k.endswith("_ms") and k not in ("total_ms", "other_ms")]
    means = [summary[k].get("mean", 0.0) for k in stage_keys]
    means.append(summary.get("other_ms", {}).get("mean", 0.0))
    stage_labels = [k.replace("_ms", "") for k in stage_keys] + ["other"]

    # Stacked horizontal bar
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    cmap = plt.get_cmap("tab10")
    left = 0.0
    total = sum(m for m in means if m > 0)
    if total <= 0:
        return
    for i, (lab, m) in enumerate(zip(stage_labels, means)):
        if m <= 0:
            continue
        ax.barh([0], [m], left=left, color=cmap(i % 10), edgecolor="black",
                linewidth=0.4, label=f"{lab} ({100 * m / total:.1f}%)")
        left += m
    ax.set_yticks([])
    ax.set_xlabel("ms")
    ax.set_title("E0.8a — Per-stage latency breakdown (mean)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.6),
              ncol=4, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_breakdown_bar")

    # Pie
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    valid = [(l, m) for l, m in zip(stage_labels, means) if m > 0]
    labels = [v[0] for v in valid]
    sizes = [v[1] for v in valid]
    colors = [cmap(i % 10) for i in range(len(valid))]
    wedges, _texts, _autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                        autopct="%1.1f%%", textprops={"fontsize": 8},
                                        wedgeprops={"linewidth": 0.6, "edgecolor": "white"})
    ax.set_title("E0.8a — Latency composition")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_breakdown_pie")

    # Total latency distribution (boxplot)
    totals = [r.get("total_ms", 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.boxplot([totals], showfliers=False, widths=0.5,
               medianprops=dict(color="#c14b4b", linewidth=1.2))
    ax.set_xticks([1])
    ax.set_xticklabels(["total"])
    ax.set_ylabel("ms")
    ax.set_title("E0.8a — Total latency distribution")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_total_latency_dist")


def _plot_e0_8b(summary: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    bars = [
        ("parameters", float(summary.get("parameters_mb", 0.0))),
        ("KV cache", float(summary.get("kv_cache_total_mb", {}).get("mean", 0.0))),
        ("activations (est.)",
         float(summary.get("activation_mb_estimate", {}).get("mean", 0.0))),
        ("peak alloc",
         float(summary.get("peak_allocated_mb", {}).get("mean", 0.0))),
        ("peak reserved",
         float(summary.get("peak_reserved_mb", {}).get("mean", 0.0))),
    ]
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    cmap = plt.get_cmap("tab10")
    ax.bar(range(len(labels)), values,
           color=[cmap(i % 10) for i in range(len(labels))],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("MB")
    ax.set_title("E0.8b — GPU memory components (batch=1)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_memory_breakdown")


def _plot_e0_8c(summary: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    per = summary.get("per_batch", {})
    if not per:
        return
    Bs = sorted(int(k) for k in per.keys())
    lat = [per[B]["latency_s"]["median"] for B in Bs]
    lat_p10 = [per[B]["latency_s"]["p10"] for B in Bs]
    lat_p90 = [per[B]["latency_s"]["p90"] for B in Bs]
    thr = [per[B]["throughput_samples_per_s"]["median"] for B in Bs]
    mem = [per[B]["peak_alloc_mb"]["median"] for B in Bs]
    eff = [summary.get("scaling_efficiency", {}).get(B, 0.0) for B in Bs]

    # Latency vs batch
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.errorbar(Bs, lat,
                yerr=[np.subtract(lat, lat_p10), np.subtract(lat_p90, lat)],
                fmt="o-", color="#3b75af", capsize=3)
    if lat:
        ideal = [lat[0] for _ in Bs]  # ideal: latency stays constant = perfect scaling
        ax.plot(Bs, ideal, "--", color="grey", lw=0.8, label="ideal (latency flat)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (s)")
    ax.set_title("E0.8c — Latency vs batch size")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_latency_vs_batch")

    # Throughput vs batch
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(Bs, thr, "o-", color="#3b8b58", label="measured")
    if thr:
        ax.plot(Bs, [thr[0] * B for B in Bs], "--", color="grey", lw=0.8, label="linear")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("E0.8c — Throughput vs batch size")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_throughput_vs_batch")

    # Memory vs batch
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(Bs, mem, "o-", color="#c14b4b")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak alloc (MB)")
    ax.set_title("E0.8c — Memory vs batch size")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_memory_vs_batch")

    # Efficiency
    if eff and any(e for e in eff):
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        ax.plot(Bs, eff, "o-", color="#7a4dab")
        ax.axhline(1.0, ls="--", color="grey", lw=0.8, label="perfect scaling")
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Scaling efficiency")
        ax.set_title("E0.8c — Scaling efficiency  (throughput / B·throughput₁)")
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_fig(fig, out_dir / "plot_efficiency_vs_batch")


def _plot_e0_8d(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    steps_rows = [r for r in rows if r.get("axis") == "diffusion_steps"]
    res_rows = [r for r in rows if r.get("axis") == "resolution"]

    if steps_rows:
        steps = [int(r["value"]) for r in steps_rows]
        lat = [r["latency_s_mean"] for r in steps_rows]
        sigma = [r["action_std_mean"] for r in steps_rows]
        peak = [r["peak_alloc_mb_mean"] for r in steps_rows]
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharex=True)
        axes[0].plot(steps, lat, "o-", color="#3b75af", label="latency (s)")
        axes[0].set_xlabel("DiT steps"); axes[0].set_ylabel("latency (s)")
        axes[0].set_title("Latency vs steps")
        ax2 = axes[0].twinx()
        ax2.plot(steps, peak, "s--", color="#c14b4b", label="peak MB")
        ax2.set_ylabel("peak alloc (MB)")
        axes[1].plot(steps, sigma, "o-", color="#7a4dab")
        axes[1].set_xlabel("DiT steps"); axes[1].set_ylabel(r"$\sigma_{\mathrm{seed}}$")
        axes[1].set_title("Action variance vs steps")
        fig.suptitle("E0.8d — Diffusion-step scaling", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save_fig(fig, out_dir / "plot_steps_scaling")

    if res_rows:
        labels = [r["value"] for r in res_rows]
        lat = [r["latency_s_mean"] for r in res_rows]
        sigma = [r["action_std_mean"] for r in res_rows]
        peak = [r["peak_alloc_mb_mean"] for r in res_rows]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
        axes[0].bar(x, lat, color="#3b75af", edgecolor="black", linewidth=0.4)
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("latency (s)")
        axes[0].set_title("Latency vs resolution")
        axes[1].bar(x, peak, color="#c14b4b", edgecolor="black", linewidth=0.4)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("peak alloc (MB)")
        axes[1].set_title("Memory vs resolution")
        fig.suptitle("E0.8d — Image-resolution scaling", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save_fig(fig, out_dir / "plot_resolution_scaling")


def _plot_e0_8e(rows: list[dict[str, Any]], out_dir: Path,
                batch_latencies: dict[int, float]) -> None:
    import matplotlib.pyplot as plt
    if not rows:
        return
    labels = [r["block_size"] for r in rows]
    n_blocks = [r["num_blocks"] for r in rows]
    seq = [r["sequential_total_s"] / 60.0 for r in rows]
    best = [r["best_total_s"] / 60.0 for r in rows]
    speedup = [r["speedup_vs_sequential"] for r in rows]
    x = np.arange(len(labels))

    # Projected total cost (minutes)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.bar(x - 0.18, seq, width=0.36, color="#c14b4b",
           edgecolor="black", linewidth=0.4, label="sequential")
    ax.bar(x + 0.18, best, width=0.36, color="#3b8b58",
           edgecolor="black", linewidth=0.4, label="best batched")
    for i, n in enumerate(n_blocks):
        ax.text(x[i], max(seq[i], best[i]) * 1.02, f"{n} blocks",
                ha="center", va="bottom", fontsize=7, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Projected total time (min)")
    ax.set_title("E0.8e — Perturbation cost projection per example")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_perturbation_projection")

    # Speedup vs block size
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(x, speedup, "o-", color="#7a4dab")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.axhline(1.0, ls="--", color="grey", lw=0.8)
    ax.set_ylabel(r"Speedup ($t_{\mathrm{seq}} / t_{\mathrm{best}}$)")
    ax.set_title("E0.8e — Batched perturbation speedup")
    fig.tight_layout()
    _save_fig(fig, out_dir / "plot_perturbation_speedup")


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
        "p95": float(np.quantile(arr, 0.95)),
    }


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


def _parse_block_sizes(spec: str) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 3:
            raise ValueError(f"block_size must be TxHxW (got {token!r})")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def _parse_resolutions(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 2:
            raise ValueError(f"resolution must be HxW (got {token!r})")
        out.append((int(parts[0]), int(parts[1])))
    return out


def _build_combined_report(out_dir: Path,
                           ex_ids: list[str],
                           a_summary: dict[str, Any],
                           b_summary: dict[str, Any],
                           c_summary: dict[str, Any],
                           d_summary: dict[str, Any],
                           e_summary: dict[str, Any]) -> None:
    def fmt_ms(d: dict[str, Any] | None, key: str = "median") -> str:
        if not d:
            return "n/a"
        return f"{d.get(key, float('nan')):.2f}"

    lines = ["# Stage 0 / E0.8 — Efficiency Profile",
             "",
             f"- Examples: {len(ex_ids)} ({', '.join(ex_ids)})",
             "",
             "## E0.8a — Latency breakdown (ms, batch=1)",
             "",
             "| Stage | mean | median | p90 | p95 |",
             "|-------|------|--------|-----|-----|"]
    for k in sorted([k for k in a_summary.keys() if k.endswith("_ms")]):
        s = a_summary.get(k, {})
        lines.append(f"| `{k}` | {s.get('mean', float('nan')):.2f} | "
                     f"{s.get('median', float('nan')):.2f} | "
                     f"{s.get('p90', float('nan')):.2f} | "
                     f"{s.get('p95', float('nan')):.2f} |")
    lines += [
        "",
        f"_Stages accounted for `{a_summary.get('accounted_fraction_mean', 0.0) * 100:.1f}%` of total wall time on average._",
        "",
        "Plots: `e0_8a_latency_breakdown/plot_breakdown_*.png`, `plot_total_latency_dist.png`",
        "",
        "## E0.8b — GPU memory (MB, batch=1)",
        "",
        "| Component | mean | median | p90 |",
        "|-----------|------|--------|-----|",
        f"| parameters | {b_summary.get('parameters_mb', 0.0):.1f} | — | — |",
        f"| KV cache total | "
        f"{b_summary.get('kv_cache_total_mb', {}).get('mean', 0.0):.1f} | "
        f"{b_summary.get('kv_cache_total_mb', {}).get('median', 0.0):.1f} | "
        f"{b_summary.get('kv_cache_total_mb', {}).get('p90', 0.0):.1f} |",
        f"| activations (est.) | "
        f"{b_summary.get('activation_mb_estimate', {}).get('mean', 0.0):.1f} | "
        f"{b_summary.get('activation_mb_estimate', {}).get('median', 0.0):.1f} | "
        f"{b_summary.get('activation_mb_estimate', {}).get('p90', 0.0):.1f} |",
        f"| peak allocated | "
        f"{b_summary.get('peak_allocated_mb', {}).get('mean', 0.0):.1f} | "
        f"{b_summary.get('peak_allocated_mb', {}).get('median', 0.0):.1f} | "
        f"{b_summary.get('peak_allocated_mb', {}).get('p90', 0.0):.1f} |",
        f"| peak reserved | "
        f"{b_summary.get('peak_reserved_mb', {}).get('mean', 0.0):.1f} | "
        f"{b_summary.get('peak_reserved_mb', {}).get('median', 0.0):.1f} | "
        f"{b_summary.get('peak_reserved_mb', {}).get('p90', 0.0):.1f} |",
        "",
        "Plot: `e0_8b_memory/plot_memory_breakdown.png`",
        "",
        "## E0.8c — Batch-size scaling",
        "",
        "| Batch | latency median (s) | throughput (samples/s) | peak alloc (MB) | scaling efficiency |",
        "|------:|-------------------:|-----------------------:|----------------:|-------------------:|",
    ]
    per_batch = c_summary.get("per_batch", {}) or {}
    eff = c_summary.get("scaling_efficiency", {}) or {}
    for B in sorted(int(k) for k in per_batch.keys()):
        r = per_batch[B]
        lines.append(
            f"| {B} | {r['latency_s']['median']:.3f} | "
            f"{r['throughput_samples_per_s']['median']:.2f} | "
            f"{r['peak_alloc_mb']['median']:.0f} | "
            f"{eff.get(B, 0.0):.2f} |"
        )
    if c_summary.get("notes"):
        lines.append("")
        lines.append("**Notes:**")
        for B, msg in (c_summary.get("notes") or {}).items():
            lines.append(f"- B={B}: {msg}")
    lines += [
        "",
        "Plots: `e0_8c_batch_scaling/plot_*.png`",
        "",
        "## E0.8d — Diffusion-step / resolution scaling",
        "",
        "Plots: `e0_8d_horizon_resolution/plot_steps_scaling.png`, "
        "`plot_resolution_scaling.png`",
        "",
        "## E0.8e — Perturbation-cost projection",
        "",
    ]
    rows = e_summary.get("rows", []) or []
    if rows:
        lines += ["| block_size | num_blocks | seq (min) | best (min) | best B | speedup |",
                  "|------------|-----------:|----------:|-----------:|-------:|--------:|"]
        for r in rows:
            lines.append(
                f"| {r['block_size']} | {r['num_blocks']} | "
                f"{r['sequential_total_s'] / 60:.1f} | "
                f"{r['best_total_s'] / 60:.1f} | "
                f"{r['best_batch']} | "
                f"{r['speedup_vs_sequential']:.2f}× |"
            )
    lines += [
        "",
        "Plots: `e0_8e_perturbation_cost/plot_perturbation_projection.png`, "
        "`plot_perturbation_speedup.png`",
        "",
        "## Recommended defaults for downstream stages",
        "",
        "- Batch perturbations at the largest **B** with `scaling_efficiency ≥ 0.7`.",
        "- Use the block size that minimises `best_total_s` while still hitting the "
        "target spatial resolution for your causal map.",
        "- Treat `e0_8e_perturbation_cost/per_block_size.csv` as the budget table when "
        "deciding Stage 1 sweep parameters.",
        "",
    ]
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
    p.add_argument("--num_examples", type=int, default=3)
    p.add_argument("--num_warmup", type=int, default=2,
                   help="Warmup forward passes before measurement (per example/config).")
    p.add_argument("--num_runs", type=int, default=8,
                   help="Measurement runs per example for E0.8a / E0.8b.")
    p.add_argument("--batch_sizes", default="1,2,4,8",
                   help="Comma-separated batch sizes to sweep in E0.8c.")
    p.add_argument("--num_seeds_ablation", type=int, default=4,
                   help="Seeds per setting for E0.8d.")
    p.add_argument("--diffusion_steps", default="1,2,4,8",
                   help="Comma-separated DiT step counts for E0.8d.")
    p.add_argument("--resolutions", default="",
                   help="Comma-separated HxW resolutions for E0.8d (empty = skip).")
    p.add_argument("--block_sizes", default="1x1x1,1x2x2,1x4x4,1x8x8,1x16x16",
                   help="Candidate block sizes (TxHxW) for E0.8e perturbation projection.")
    p.add_argument("--num_seeds_for_significance", type=int, default=2,
                   help="Seed repetition factor used to project Stage-1 budget in E0.8e.")
    p.add_argument("--skip_e0_8a", action="store_true")
    p.add_argument("--skip_e0_8b", action="store_true")
    p.add_argument("--skip_e0_8c", action="store_true")
    p.add_argument("--skip_e0_8d", action="store_true")
    p.add_argument("--skip_e0_8e", action="store_true")
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
    for entry in chosen:
        try:
            obs = build_obs(entry)
        except Exception as e:
            logger.warning("Skipping %s: %s", entry.get("example_id"), e)
            continue
        examples.append((entry["example_id"], obs))
    if not examples:
        logger.error("No usable examples.")
        sys.exit(1)
    ex_ids = [eid for eid, _ in examples]
    logger.info("Profiling on %d examples: %s", len(ex_ids), ex_ids)

    a_summary: dict[str, Any] = {}
    b_summary: dict[str, Any] = {}
    c_summary: dict[str, Any] = {}
    d_summary: dict[str, Any] = {}
    e_summary: dict[str, Any] = {}

    if not args.skip_e0_8a:
        logger.info("E0.8a — latency breakdown ...")
        a_summary = run_e0_8a(policy, examples,
                              args.num_warmup, args.num_runs,
                              out_dir / "e0_8a_latency_breakdown")

    if not args.skip_e0_8b:
        logger.info("E0.8b — memory ...")
        b_summary = run_e0_8b(policy, examples,
                              args.num_runs, out_dir / "e0_8b_memory")

    if not args.skip_e0_8c:
        Bs = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
        logger.info("E0.8c — batch-size scaling: %s", Bs)
        c_summary = run_e0_8c(policy, examples, sorted(set(Bs)),
                              args.num_warmup, args.num_runs,
                              out_dir / "e0_8c_batch_scaling")

    if not args.skip_e0_8d:
        steps = sorted({int(x) for x in args.diffusion_steps.split(",") if x.strip()})
        res = _parse_resolutions(args.resolutions) if args.resolutions else []
        logger.info("E0.8d — diffusion_steps=%s resolutions=%s", steps, res)
        d_summary = run_e0_8d(policy, examples, steps, res,
                              args.num_seeds_ablation,
                              out_dir / "e0_8d_horizon_resolution")

    if not args.skip_e0_8e:
        block_sizes = _parse_block_sizes(args.block_sizes)
        logger.info("E0.8e — perturbation projection over %d block sizes ...", len(block_sizes))
        e_summary = run_e0_8e(policy, examples, block_sizes,
                              a_summary, c_summary,
                              args.num_seeds_for_significance,
                              out_dir / "e0_8e_perturbation_cost")

    _build_combined_report(out_dir, ex_ids, a_summary, b_summary,
                           c_summary, d_summary, e_summary)
    (out_dir / "combined_report.json").write_text(json.dumps({
        "examples": ex_ids,
        "e0_8a": a_summary,
        "e0_8b": b_summary,
        "e0_8c": c_summary,
        "e0_8d": d_summary,
        "e0_8e": e_summary,
    }, indent=2))

    # Paper-ready compact CSV
    with (out_dir / "baseline_efficiency.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        if a_summary.get("total_ms"):
            for q in ("mean", "median", "p90", "p95"):
                w.writerow([f"latency_{q}", f"{a_summary['total_ms'].get(q, float('nan')):.2f}", "ms"])
        if b_summary.get("peak_allocated_mb"):
            w.writerow(["peak_alloc_mean", f"{b_summary['peak_allocated_mb']['mean']:.1f}", "MB"])
            w.writerow(["parameters_mb", f"{b_summary.get('parameters_mb', 0.0):.1f}", "MB"])
            w.writerow(["kv_cache_mean_mb",
                        f"{b_summary.get('kv_cache_total_mb', {}).get('mean', 0.0):.1f}", "MB"])
        if c_summary.get("scaling_efficiency"):
            for B, eff in c_summary["scaling_efficiency"].items():
                w.writerow([f"scaling_efficiency_B{B}", f"{eff:.2f}", "ratio"])
        if e_summary.get("rows"):
            for r in e_summary["rows"]:
                w.writerow([f"projected_min_block_{r['block_size']}",
                            f"{r['best_total_s'] / 60.0:.2f}", "min"])

    logger.info("Done. Combined report -> %s", out_dir / "combined_report.md")


if __name__ == "__main__":
    main()
