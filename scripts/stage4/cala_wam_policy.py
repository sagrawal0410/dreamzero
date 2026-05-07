#!/usr/bin/env python3
"""Stage 4 — CALA-WAM policy adapter for live closed-loop sim eval.

Wraps `GrootSimPolicy` so that every inference call is preceded by a
CALA-WAM retention step on `head.ys`. Designed to drop into the existing
websocket inference server (`socket_test_optimized_AR.py`) so that
`eval_utils/run_sim_eval.py` can run real DROID Isaac-Lab rollouts with
CALA-WAM allocation enabled.

Three importance-providing strategies, picked by `--strategy`:

    cached     : load a precomputed Stage-1 map for the *first observed
                 example_id* and reuse it across the whole episode.
                 Lowest overhead; assumes a cache for the start state.
    attention  : run one cheap baseline forward, take the per-token norm
                 of the last DiT block as the importance map (Stage-2 proxy).
                 No precompute, but requires one extra forward per call.
    dynamic    : compute a fresh action-causal map every K steps via a
                 small perturbation sweep (expensive; reference-only).

Usage
-----

This file exposes a `CalaWamPolicy` class that follows the
`groot.vla.model.n1_5.sim_policy.GrootSimPolicy` API surface used by the
existing servers (`forward`, `reset`, `lazy_joint_forward_causal`). Wiring
into the AR-DROID server is intentionally minimal — replace the
`policy = GrootSimPolicy(...)` line in `socket_test_optimized_AR.py` with:

    from scripts.stage4.cala_wam_policy import CalaWamPolicy
    policy = CalaWamPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        strategy="attention",
        budget_pct=50.0,
        variant="A_hard_retention",
    )

The class delegates everything else (transforms, distributed wiring, action
unapply) to the wrapped `GrootSimPolicy`. Only `lazy_joint_forward_causal`
is intercepted to apply the CALA-WAM mask.

This file is a reference adapter; running real Isaac-Lab DROID rollouts
additionally requires the `sim_evals` package set up per the project
README. Use `scripts/stage4/closed_loop_compute.py` for the offline
trajectory-replay metric reported in the paper tables.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("stage4.cala_wam_policy")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

from tianshou.data import Batch  # noqa: E402

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
STAGE3_DIR = SCRIPT_DIR.parent / "stage3"
sys.path.insert(0, str(STAGE0_DIR))
sys.path.insert(0, str(STAGE3_DIR))

import perturbation_suite as ps  # noqa: E402
from allocation_compute import _build_perturbed_ys, _load_method_map  # noqa: E402


# ============================================================================
# Importance providers
# ============================================================================


def _resolve(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


class _DiTAttentionHook:
    def __init__(self) -> None:
        self.last_output: torch.Tensor | None = None

    def __call__(self, _mod, _inputs, outputs):
        out = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
        if torch.is_tensor(out):
            self.last_output = out.detach().to("cpu")


class CachedImportanceProvider:
    """Loads a precomputed Stage-1 heatmap from `--cache_dir` once and reuses it."""

    def __init__(self, cache_dir: Path, primary_op: str, primary_metric: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.primary_op = primary_op
        self.primary_metric = primary_metric
        self._cached_map: np.ndarray | None = None
        self._cached_for: str | None = None

    def get(self, ys_shape: tuple[int, int, int, int, int], obs: dict[str, Any]) -> np.ndarray | None:
        T, H, W = int(ys_shape[2]), int(ys_shape[3]), int(ys_shape[4])
        if self._cached_map is not None:
            return self._cached_map
        # Search for any heatmap under cache_dir; pick the first available one.
        if not self.cache_dir.exists():
            return None
        for sub in sorted(self.cache_dir.iterdir()):
            if not sub.is_dir():
                continue
            npz = sub / "heatmaps.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            key = f"{self.primary_op}__{self.primary_metric}"
            if key not in z.files:
                continue
            arr = z[key].astype(np.float32)
            # Resize if shape mismatches the live latent grid.
            if arr.shape != (T, H, W):
                import cv2
                arr2 = np.nanmean(arr, axis=0) if arr.ndim == 3 else arr
                arr2 = cv2.resize(arr2, (W, H), interpolation=cv2.INTER_LINEAR)
                arr = np.broadcast_to(arr2[None, :, :], (T, H, W)).copy()
            self._cached_map = arr
            self._cached_for = sub.name
            logger.info("Cached importance loaded from %s", sub.name)
            return arr
        return None


class AttentionImportanceProvider:
    """Uses the last-DiT-block output (captured by `CalaWamPolicy` during the
    baseline forward) as the importance map.

    Set `last_attention` to the captured tensor before calling `get()`.
    """

    def __init__(self, head: Any) -> None:
        self.head = head
        self.last_attention: torch.Tensor | None = None

    def get(self, ys_shape: tuple[int, int, int, int, int], obs: dict[str, Any]) -> np.ndarray | None:
        T, H, W = int(ys_shape[2]), int(ys_shape[3]), int(ys_shape[4])
        out = self.last_attention
        if out is None or not torch.is_tensor(out) or out.ndim != 3:
            return None
        # Reshape: WAN DiT typically uses patch (1,2,2) → N = T * (H/2) * (W/2)
        norms = torch.linalg.vector_norm(out[0].float(), dim=-1).numpy()  # (N,)
        h_d = max(1, H // 2); w_d = max(1, W // 2)
        if T * h_d * w_d == norms.size:
            arr = norms.reshape(T, h_d, w_d)
        else:
            per_t = norms.size // max(1, T)
            side = int(round(math.sqrt(max(1, per_t))))
            if side * side != per_t:
                return None
            arr = norms[:T * side * side].reshape(T, side, side)
        import cv2
        out_resized = np.stack(
            [cv2.resize(arr[t].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
             for t in range(T)], axis=0)
        return out_resized


# ============================================================================
# CALA-WAM-wrapped policy
# ============================================================================


class CalaWamPolicy(GrootSimPolicy):
    """`GrootSimPolicy` + CALA-WAM retention applied to `head.ys` per call.

    Constructor arguments are a superset of `GrootSimPolicy.__init__`. Extra args:

        strategy        : "cached" | "attention" | "dynamic"
        budget_pct      : retention budget (e.g. 50.0)
        variant         : "A_hard_retention" | "B_soft_fidelity" | "C_temporal_cache"
                          | "D_global_summary"
        cache_dir       : Stage-1 maps directory (required for strategy="cached")
        primary_op      : Stage-1 operator name (default "local_mean")
        primary_metric  : Stage-1 metric name (default "action_l2")
        bypass_first_calls : skip CALA-WAM for the first N calls (warmup)
    """

    def __init__(self,
                 *args: Any,
                 strategy: str = "attention",
                 budget_pct: float = 50.0,
                 variant: str = "A_hard_retention",
                 cache_dir: str | None = None,
                 primary_op: str = "local_mean",
                 primary_metric: str = "action_l2",
                 bypass_first_calls: int = 0,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cala_strategy = strategy
        self._cala_budget = float(budget_pct)
        self._cala_variant = variant
        self._cala_call_count = 0
        self._cala_bypass = int(bypass_first_calls)
        self._cala_primary_op = primary_op
        self._cala_primary_metric = primary_metric

        head = self.trained_model.action_head
        self._cala_provider: Any
        if strategy == "cached":
            if not cache_dir:
                raise ValueError("strategy='cached' requires cache_dir")
            self._cala_provider = CachedImportanceProvider(
                Path(cache_dir), primary_op, primary_metric
            )
        elif strategy == "attention":
            self._cala_provider = AttentionImportanceProvider(head)
        elif strategy == "dynamic":
            self._cala_provider = None  # placeholder; falls back to attention
            logger.warning("strategy='dynamic' is not yet implemented; falling back to 'attention'.")
            self._cala_provider = AttentionImportanceProvider(head)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        logger.info("CalaWamPolicy: strategy=%s budget_pct=%g variant=%s",
                    strategy, budget_pct, variant)

    # ---------------------------------------------------------------- forward
    def lazy_joint_forward_causal(self, batch: Batch, video: Any | None = None,
                                  latent_video: Any | None = None,
                                  state: Any | None = None,
                                  video_only: bool = False, **kwargs: Any):
        # Bypass CALA-WAM during warmup or whenever budget == 100%.
        self._cala_call_count += 1
        if self._cala_call_count <= self._cala_bypass or self._cala_budget >= 100.0 - 1e-6:
            return super().lazy_joint_forward_causal(
                batch, video=video, latent_video=latent_video,
                state=state, video_only=video_only, **kwargs,
            )

        # We need the baseline ys and clip_feas to apply retention. Use the
        # same monkey-patch trick as Stage-0 perturbation_suite. If the
        # attention strategy is active, also install a hook on the last DiT
        # block to capture activations during the baseline forward.
        head = self.trained_model.action_head
        cache: dict[str, Any] = {}
        original = head.encode_image

        def capturing(image, num_frames, height, width):
            clip, ys, new_image = original(image, num_frames, height, width)
            cache["clip_feas"] = clip.detach().clone()
            cache["ys"] = ys.detach().clone()
            cache["new_image"] = new_image.detach().clone() if torch.is_tensor(new_image) else new_image
            return clip, ys, new_image

        attn_hook = None
        attn_handle = None
        if isinstance(self._cala_provider, AttentionImportanceProvider):
            blocks = _resolve(head, "model.blocks")
            if blocks is not None and len(blocks) > 0:
                attn_hook = _DiTAttentionHook()
                attn_handle = blocks[len(blocks) - 1].register_forward_hook(attn_hook)

        try:
            head.encode_image = capturing
            t0 = time.perf_counter()
            with torch.inference_mode():
                _result, _video_pred = super().lazy_joint_forward_causal(
                    batch, video=video, latent_video=latent_video,
                    state=state, video_only=video_only, **kwargs,
                )
            base_s = time.perf_counter() - t0
        finally:
            head.encode_image = original
            if attn_handle is not None:
                attn_handle.remove()
            if isinstance(self._cala_provider, AttentionImportanceProvider) and attn_hook is not None:
                self._cala_provider.last_attention = attn_hook.last_output

        ys_base = cache.get("ys")
        if ys_base is None:
            return super().lazy_joint_forward_causal(
                batch, video=video, latent_video=latent_video,
                state=state, video_only=video_only, **kwargs,
            )

        # Acquire importance map
        importance = None
        try:
            importance = self._cala_provider.get(ys_base.shape, batch.obs)
        except Exception as e:
            logger.warning("Importance provider failed: %s", e)
            importance = None
        if importance is None:
            logger.info("Importance unavailable; running baseline forward.")
            return super().lazy_joint_forward_causal(
                batch, video=video, latent_video=latent_video,
                state=state, video_only=video_only, **kwargs,
            )

        # Apply retention/compression and run a perturbed forward
        z_ch_start = 4 if ys_base.shape[1] > 4 else 0
        ys_perturbed = _build_perturbed_ys(
            self._cala_variant, ys_base.clone(), importance.astype(np.float32),
            self._cala_budget, z_ch_start, prev_ys=None,
        )

        # Replay forward with patched encode_image returning ys_perturbed
        def returning(image, num_frames, height, width):
            ni = cache["new_image"]
            ni_ret = ni.clone() if torch.is_tensor(ni) else ni
            return cache["clip_feas"].clone(), ys_perturbed, ni_ret

        try:
            head.encode_image = returning
            return super().lazy_joint_forward_causal(
                batch, video=video, latent_video=latent_video,
                state=state, video_only=video_only, **kwargs,
            )
        finally:
            head.encode_image = original


# ============================================================================
# Standalone CLI: smoke-test the wrapper on a manifest entry
# ============================================================================


def _smoke() -> None:
    import argparse
    SCRIPT0 = Path(__file__).resolve().parent.parent / "stage0"
    sys.path.insert(0, str(SCRIPT0))
    from _common import build_obs, init_dist_single_process, EMBODIMENT_TAG  # type: ignore

    p = argparse.ArgumentParser(description="Smoke-test CalaWamPolicy on one manifest entry.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True)
    p.add_argument("--example_index", type=int, default=0)
    p.add_argument("--strategy", default="attention",
                   choices=["cached", "attention", "dynamic"])
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--budget_pct", type=float, default=50.0)
    p.add_argument("--variant", default="A_hard_retention")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import json
    with Path(args.task_suite).open("r") as f:
        manifest = json.load(f)
    if not manifest:
        print("Manifest empty."); sys.exit(1)
    entry = manifest[max(0, min(args.example_index, len(manifest) - 1))]

    init_dist_single_process()
    policy = CalaWamPolicy(
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        model_path=args.checkpoint,
        device=args.device,
        strategy=args.strategy,
        budget_pct=args.budget_pct,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )
    obs = build_obs(entry)
    t0 = time.perf_counter()
    result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
    elapsed = time.perf_counter() - t0
    print(f"OK  example={entry['example_id']}  forward={elapsed:.3f}s")
    for k in dir(result.act):
        if k.startswith("action."):
            v = getattr(result.act, k)
            shape = v.shape if hasattr(v, "shape") else None
            print(f"  {k}: shape={shape}")


if __name__ == "__main__":
    _smoke()
