"""Shared helpers for Stage 0 scripts (E0.0 / E0.1 / E0.2 / E0.3).

Kept intentionally tiny so each Stage 0 script can stay self-contained while
sharing the small surface that touches the model: distributed init, the
DROID observation schema, the per-example observation builder, and the
causal-state reset between examples.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------
# Modality keys for OXE_DROID — must match training config
# ----------------------------------------------------------------------------

VIDEO_KEYS: tuple[str, ...] = (
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
)
LANGUAGE_KEY: str = "annotation.language.action_text"
ACTION_KEYS: tuple[str, ...] = ("action.joint_position", "action.gripper_position")
EMBODIMENT_TAG: str = "oxe_droid"


def init_dist_single_process() -> None:
    """Initialise a 1-rank gloo process group iff none exists yet."""
    import torch.distributed as dist
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


def read_frame(mp4_path: str, frame_index: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {mp4_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_obs(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a single-frame observation dict matching what GrootSimPolicy expects.

    First-call AR_droid format: T=1 frame per camera, state is (1, D), prompt
    is a Python string under LANGUAGE_KEY.
    """
    obs: dict[str, Any] = {}
    for key in VIDEO_KEYS:
        mp4 = entry["video_paths"].get(key)
        if mp4 is None:
            raise KeyError(f"Manifest entry missing video for {key}: {entry['example_id']}")
        frame = read_frame(mp4, entry["frame_index"])
        obs[key] = frame[np.newaxis, ...].astype(np.uint8)

    state_vec = np.asarray(entry["state_at_t"], dtype=np.float64).reshape(-1)
    if state_vec.size >= 8:
        obs["state.joint_position"] = state_vec[:7].reshape(1, -1).astype(np.float64)
        obs["state.gripper_position"] = state_vec[7:8].reshape(1, -1).astype(np.float64)
    else:
        obs["state.joint_position"] = state_vec[:7].reshape(1, -1).astype(np.float64)
        obs["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

    obs[LANGUAGE_KEY] = entry.get("instruction", "") or ""
    return obs


def reset_causal_state(policy: Any) -> None:
    """Reset DreamZero's causal action-head state so each example starts fresh."""
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


def flatten_action_dict(actions: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate per-key actions on the channel dim into one (H, A) chunk."""
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


def configure_neurips_matplotlib() -> None:
    """Set matplotlib rcParams for clean, paper-ready figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (import for side effects)
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.4,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
