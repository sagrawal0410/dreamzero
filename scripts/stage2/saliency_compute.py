#!/usr/bin/env python3
"""Stage 2 / Compute — Saliency-proxy maps for the comparison study (E2.1).

Computes, for every example in the Stage-0 manifest, a stack of *baseline*
saliency proxies that downstream analysis (E2.1 IoU + correlation tables,
Figure 2 gallery) and Stage-3 budget allocation reuse:

    center            : analytic 2-D Gaussian centred on the latent grid.
    gripper_proxy     : analytic 2-D Gaussian biased to bottom-centre
                         (where the end-effector usually lies in DROID views).
    edges_input       : Sobel-magnitude of the input camera frame, downsampled.
    clip_features     : norm of the CLIP image-encoder spatial-token output.
    dit_attention     : norm of the last WAN-DiT transformer block's output
                         per spatial token, reshaped to a (T_lat, H_lat, W_lat)
                         "what the model computed strongly" map.
    optical_flow      : per-pixel optical-flow magnitude of the VAE-decoded
                         baseline future video, downsampled to the latent grid.

All maps are written at the **latent resolution** so they can be compared to
the Stage-1 action-causal heatmap directly. The script also writes the
input camera frame and the decoded baseline future video for Figure 2.

For each example we additionally record:
    - clip_feas / last_dit_block raw shapes (for reproducibility)
    - input image resolution (for resizing in Stage 2 / Stage 3)

Example:

    python scripts/stage2/saliency_compute.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_examples 12 \\
        --output_dir runs/stage2_saliency

Smoke (≈3 min):

    python scripts/stage2/saliency_compute.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --num_examples 2 \\
        --output_dir runs/stage2_saliency_smoke

Per-example output:

    runs/stage2_saliency/<example_id>/
        saliency_maps.npz   - {center, gripper_proxy, edges_input, clip_features,
                                dit_attention, optical_flow} 3-D arrays in latent shape
        baseline_input.png
        baseline_future.mp4   - decoded baseline future video (optical-flow source)
        meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2.compute")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True

from tianshou.data import Batch  # noqa: E402

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
sys.path.insert(0, str(STAGE0_DIR))

from _common import (  # noqa: E402
    EMBODIMENT_TAG,
    build_obs,
    init_dist_single_process,
    read_frame,
    reset_causal_state,
)


# ============================================================================
# Hook helpers — capture activations during a single baseline forward
# ============================================================================


@torch._dynamo.disable
def _resolve(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


class _CaptureHook:
    """Record the first call's input/output of a module."""
    def __init__(self) -> None:
        self.inputs: tuple[torch.Tensor, ...] = ()
        self.output: torch.Tensor | None = None
        self.captured = False

    def __call__(self, _mod, inputs, outputs):
        if self.captured:
            return
        try:
            if isinstance(inputs, (tuple, list)) and inputs:
                self.inputs = tuple(
                    x.detach().to("cpu") if torch.is_tensor(x) else x for x in inputs
                )
            out = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
            if torch.is_tensor(out):
                self.output = out.detach().to("cpu")
        finally:
            self.captured = True


def _install_capture_hooks(head: Any) -> tuple[dict[str, _CaptureHook], list[Any]]:
    """Install forward hooks on (image_encoder, last DiT block, full DiT)."""
    out: dict[str, _CaptureHook] = {}
    handles: list[Any] = []

    def _try_hook(name: str, mod: Any) -> None:
        if mod is None or not hasattr(mod, "register_forward_hook"):
            return
        h = _CaptureHook()
        try:
            handles.append(mod.register_forward_hook(h))
            out[name] = h
        except Exception as e:
            logger.debug("hook on %s failed: %s", name, e)

    # CLIP image encoder
    for path in ("image_encoder", "image_encoder.model"):
        _try_hook("image_encoder", _resolve(head, path)); break
    # Last DiT transformer block
    blocks = _resolve(head, "model.blocks")
    if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
        try:
            _try_hook("dit_last_block", blocks[len(blocks) - 1])
        except Exception:
            pass
    return out, handles


def _remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# ============================================================================
# Per-proxy saliency
# ============================================================================


def _gaussian_2d(shape: tuple[int, int], cy_frac: float, cx_frac: float,
                 sigma_frac: float) -> np.ndarray:
    H, W = shape
    cy = cy_frac * (H - 1); cx = cx_frac * (W - 1)
    sy = max(1.0, sigma_frac * H); sx = max(1.0, sigma_frac * W)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    g = np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2)))
    return g.astype(np.float32)


def _resize_2d(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    import cv2
    Ht, Wt = target_hw
    a = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    return cv2.resize(a, (Wt, Ht), interpolation=cv2.INTER_LINEAR)


def _broadcast_to_3d(arr2: np.ndarray, T_lat: int) -> np.ndarray:
    """Replicate a (H, W) 2-D map across T_lat → (T, H, W) array."""
    return np.broadcast_to(arr2[None, :, :], (T_lat, *arr2.shape)).copy()


def _saliency_center(latent_shape: tuple[int, int, int]) -> np.ndarray:
    T, H, W = latent_shape
    g = _gaussian_2d((H, W), 0.5, 0.5, 0.18)
    return _broadcast_to_3d(g, T)


def _saliency_gripper(latent_shape: tuple[int, int, int]) -> np.ndarray:
    T, H, W = latent_shape
    g = _gaussian_2d((H, W), 0.78, 0.5, 0.16)
    return _broadcast_to_3d(g, T)


def _saliency_edges(input_rgb: np.ndarray, latent_shape: tuple[int, int, int]) -> np.ndarray:
    import cv2
    T, H, W = latent_shape
    g = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() > 0:
        mag /= mag.max()
    mag_lat = _resize_2d(mag, (H, W))
    return _broadcast_to_3d(mag_lat, T)


def _saliency_clip(image_encoder_output: torch.Tensor | None,
                   latent_shape: tuple[int, int, int]) -> np.ndarray:
    """Per-token norm of CLIP output, reshaped to a square spatial grid."""
    T, H, W = latent_shape
    out = np.zeros((T, H, W), dtype=np.float32)
    if image_encoder_output is None:
        return out
    o = image_encoder_output.float()
    # Common CLIP shapes: (B, N_tokens, D) with N_tokens = 1 + h*w (CLS + grid).
    if o.ndim != 3:
        return out
    B, N, D = o.shape
    feat = o[0].numpy()                  # (N, D)
    norms = np.linalg.norm(feat, axis=-1)  # (N,)
    grid = norms[1:] if (N - 1) > 0 else norms
    n_grid = int(round(math.sqrt(max(1, grid.size))))
    if n_grid * n_grid != grid.size:
        # Try common rect aspect 17 × 30 etc.; if the size is composite, factor it.
        for h in range(2, int(math.sqrt(grid.size)) + 2):
            if grid.size % h == 0:
                w = grid.size // h
                arr = grid.reshape(h, w)
                lat2 = _resize_2d(arr.astype(np.float32), (H, W))
                return _broadcast_to_3d(lat2, T)
        return out
    arr = grid.reshape(n_grid, n_grid)
    lat2 = _resize_2d(arr.astype(np.float32), (H, W))
    return _broadcast_to_3d(lat2, T)


def _saliency_dit(dit_output: torch.Tensor | None,
                  latent_shape: tuple[int, int, int]) -> np.ndarray:
    """Per-token norm of last DiT block output, reshaped to (T_lat, h_dit, w_dit)
    then resized to (T_lat, H_lat, W_lat)."""
    T, H, W = latent_shape
    out = np.zeros((T, H, W), dtype=np.float32)
    if dit_output is None:
        return out
    o = dit_output.float()
    if o.ndim != 3:
        return out
    B, N, D = o.shape
    feat = o[0].numpy()                            # (N, D)
    norms = np.linalg.norm(feat, axis=-1)          # (N,)
    # WAN DiT patchifies space by (1, 2, 2) usually. So N = T_lat * (H_lat//2) * (W_lat//2).
    # Try the obvious factoring:
    h_d = max(1, H // 2); w_d = max(1, W // 2)
    if T * h_d * w_d == norms.size:
        arr = norms.reshape(T, h_d, w_d)
    else:
        # Fallback: square-ish reshape per time slice.
        per_t = norms.size // max(1, T)
        side = int(round(math.sqrt(max(1, per_t))))
        if side * side != per_t:
            return out
        arr = norms[:T * side * side].reshape(T, side, side)
    out_resized = np.stack(
        [_resize_2d(arr[t].astype(np.float32), (H, W)) for t in range(T)], axis=0
    )
    return out_resized


def _decode_baseline_video(head: Any, video_pred: torch.Tensor | None,
                           out_path: Path) -> np.ndarray | None:
    """VAE-decode the baseline future-video latents to (T, H, W, 3) uint8.
    Saves a sanity MP4 to out_path."""
    if video_pred is None or not torch.is_tensor(video_pred):
        return None
    try:
        import imageio
        from einops import rearrange
        with torch.inference_mode():
            params = next(head.vae.parameters(), None)
            if params is None:
                return None
            vp = video_pred.to(params.device, dtype=params.dtype)
            frames = head.vae.decode(
                vp,
                tiled=getattr(head, "tiled", True),
                tile_size=(getattr(head, "tile_size_height", 34),
                           getattr(head, "tile_size_width", 34)),
                tile_stride=(getattr(head, "tile_stride_height", 18),
                             getattr(head, "tile_stride_width", 16)),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames = ((frames.float() + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), list(frames), fps=5, codec="libx264")
        return frames
    except Exception as e:
        logger.warning("VAE decode of future video failed: %s", e)
        return None


def _saliency_optical_flow(future_frames: np.ndarray | None,
                           latent_shape: tuple[int, int, int]) -> np.ndarray:
    T, H, W = latent_shape
    out = np.zeros((T, H, W), dtype=np.float32)
    if future_frames is None or future_frames.shape[0] < 2:
        return out
    import cv2
    motion = np.zeros(future_frames.shape[1:3], dtype=np.float32)
    prev_gray = cv2.cvtColor(future_frames[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, future_frames.shape[0]):
        cur_gray = cv2.cvtColor(future_frames[t], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion += mag
        prev_gray = cur_gray
    if motion.max() > 0:
        motion /= motion.max()
    motion_lat = _resize_2d(motion, (H, W))
    return _broadcast_to_3d(motion_lat, T)


# ============================================================================
# Per-example computation
# ============================================================================


def compute_example(policy: GrootSimPolicy, entry: dict[str, Any],
                    out_dir: Path, seed: int = 0,
                    camera_for_input: str = "video.exterior_image_1_left",
                    decode_video_for_flow: bool = True) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    head = policy.trained_model.action_head

    # Save input camera frame
    import cv2
    rgb_path = out_dir / "baseline_input.png"
    paths = entry.get("video_paths") or {}
    cam_path = paths.get(camera_for_input) or next(iter(paths.values()), None)
    rgb_input: np.ndarray | None = None
    if cam_path:
        try:
            rgb_input = read_frame(cam_path, int(entry.get("frame_index", 0)))
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_input, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.warning("Read input frame failed: %s", e)

    # One baseline forward with hooks
    obs = build_obs(entry)
    reset_causal_state(policy)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hooks, handles = _install_capture_hooks(head)
    try:
        with torch.inference_mode():
            _result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        _remove_hooks(handles)

    ys = head.ys.detach().to("cpu") if torch.is_tensor(getattr(head, "ys", None)) else None
    if ys is None:
        raise RuntimeError(f"head.ys not set after forward for {entry['example_id']}")
    T_lat, H_lat, W_lat = int(ys.shape[2]), int(ys.shape[3]), int(ys.shape[4])
    latent_shape = (T_lat, H_lat, W_lat)
    logger.info("[%s] latent T×H×W = %d×%d×%d", entry["example_id"], T_lat, H_lat, W_lat)

    # Decode baseline future video for optical flow
    future_frames: np.ndarray | None = None
    if decode_video_for_flow:
        future_frames = _decode_baseline_video(
            head, video_pred, out_dir / "baseline_future.mp4"
        )

    # Compute proxies
    sal: dict[str, np.ndarray] = {}
    sal["center"] = _saliency_center(latent_shape)
    sal["gripper_proxy"] = _saliency_gripper(latent_shape)
    if rgb_input is not None:
        sal["edges_input"] = _saliency_edges(rgb_input, latent_shape)
    else:
        sal["edges_input"] = np.zeros(latent_shape, dtype=np.float32)
    sal["clip_features"] = _saliency_clip(hooks.get("image_encoder").output if "image_encoder" in hooks else None,
                                          latent_shape)
    sal["dit_attention"] = _saliency_dit(hooks.get("dit_last_block").output if "dit_last_block" in hooks else None,
                                         latent_shape)
    sal["optical_flow"] = _saliency_optical_flow(future_frames, latent_shape)

    np.savez_compressed(out_dir / "saliency_maps.npz", **sal)

    meta = {
        "example_id": entry["example_id"],
        "task_group": entry.get("task_group"),
        "task_name": entry.get("task_name"),
        "instruction": entry.get("instruction"),
        "episode_index": entry.get("episode_index"),
        "frame_index": entry.get("frame_index"),
        "role": entry.get("role"),
        "video_paths": paths,
        "latent_shape": [T_lat, H_lat, W_lat],
        "input_resolution": ([rgb_input.shape[0], rgb_input.shape[1]] if rgb_input is not None else None),
        "proxies": sorted(sal.keys()),
        "decoded_future_available": future_frames is not None,
        "decoded_future_shape": ([int(x) for x in future_frames.shape] if future_frames is not None else None),
        "z_channel_start": 4 if (ys is not None and ys.shape[1] > 4) else 0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


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


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_examples", type=int, default=12)
    p.add_argument("--phase_balanced", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--no_decode_video", action="store_true",
                   help="Skip decoding the future video (no optical-flow saliency).")
    p.add_argument("--camera", default="video.exterior_image_1_left",
                   help="Which camera frame to use as the input image for figures.")
    args = p.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    metas: list[dict[str, Any]] = []
    for i, entry in enumerate(chosen):
        ex_dir = out_dir / entry["example_id"]
        if args.skip_existing and (ex_dir / "saliency_maps.npz").exists():
            logger.info("[%d/%d] %s already done, skipping", i + 1, len(chosen), entry["example_id"])
            try:
                metas.append(json.loads((ex_dir / "meta.json").read_text()))
            except Exception:
                pass
            continue
        logger.info("[%d/%d] saliency proxies for %s", i + 1, len(chosen), entry["example_id"])
        try:
            m = compute_example(
                policy, entry, ex_dir, seed=args.seed,
                camera_for_input=args.camera,
                decode_video_for_flow=not args.no_decode_video,
            )
            metas.append(m)
        except Exception as e:
            logger.error("compute_example failed for %s: %s", entry["example_id"], e, exc_info=True)
            continue

    (out_dir / "compute_run.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "num_examples": len(metas),
        "examples": metas,
    }, indent=2))
    logger.info("Done. Manifest -> %s", out_dir / "compute_run.json")


if __name__ == "__main__":
    main()
