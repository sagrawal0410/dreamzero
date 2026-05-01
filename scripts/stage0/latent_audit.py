#!/usr/bin/env python3
"""Stage 0 / E0.2 — Latent Access Sanity for DreamZero-DROID.

Pinpoints exactly which latent tensors you can access and perturb. For one
example from the Stage-0 manifest, this:

  1. Registers forward hooks on the action head's VAE encoder/decoder, the
     CLIP image encoder, the T5 text encoder, the WAN DiT patch embedding,
     and a few transformer blocks (first / middle / last).
  2. Captures stored attributes set during forward (`clip_feas`, `ys`,
     encoded language).
  3. Runs ONE inference call and records every captured tensor's shape /
     dtype / spatial-temporal layout.
  4. Tries to VAE-decode the predicted future-video latents (the DiT output)
     so you can confirm spatial alignment back to pixels.
  5. Writes:

         latent_audit.json   — full structured record
         latent_audit.csv    — per-tensor table
         latent_audit.md     — paper-appendix-ready Markdown table
         decoded_video_pred.mp4 — decoded future video (sanity)

Example:

    python scripts/stage0/latent_audit.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --task_suite runs/stage0_suite/manifest.json \\
        --example_index 0 \\
        --output_dir runs/stage0_latents
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage0.latent_audit")

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
    init_dist_single_process,
    reset_causal_state,
)


# ----------------------------------------------------------------------------
# Hook record
# ----------------------------------------------------------------------------


@dataclass
class TensorRecord:
    """One row of the latent audit table."""
    name: str
    source: str
    shape: list[int]
    dtype: str
    rank: int
    spatial: bool                 # has H, W (or single spatial dim)
    temporal: bool                # has T (frame) dim
    decodable: bool               # can be decoded back to RGB via VAE
    perturbable: bool             # safe to add noise / mask in-place
    notes: str = ""

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _shape_of(t: Any) -> tuple[list[int], str]:
    if torch.is_tensor(t):
        return list(t.shape), str(t.dtype).replace("torch.", "")
    if isinstance(t, np.ndarray):
        return list(t.shape), str(t.dtype)
    if isinstance(t, (list, tuple)) and t and torch.is_tensor(t[0]):
        return list(t[0].shape), str(t[0].dtype).replace("torch.", "")
    return [], "n/a"


def _classify_layout(shape: list[int], hint: str) -> tuple[bool, bool]:
    """Heuristic: is this tensor spatial / temporal based on rank and module hint?"""
    h = hint.lower()
    rank = len(shape)
    if rank == 5:
        # B C T H W or B T H W C
        return True, True
    if rank == 4:
        # Could be B C H W (spatial) or B T H W (?)
        return True, ("vae" in h or "video" in h)
    if rank == 3:
        # B N D — token sequence; spatial if from patch_embedding or block
        return ("patch" in h or "block" in h or "dit" in h), False
    if rank == 2:
        return False, False
    return False, False


# ----------------------------------------------------------------------------
# Module discovery (best-effort)
# ----------------------------------------------------------------------------


def _resolve(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _candidate_modules(head: Any) -> list[tuple[str, Any]]:
    """Best-effort list of (module_name, module) to hook."""
    out: list[tuple[str, Any]] = []

    # VAE: WanVideoVAE / WanVideoVAE38 — both expose .encoder and .decoder
    for path in ("vae", "vae.encoder", "vae.decoder", "vae.model.encoder", "vae.model.decoder"):
        m = _resolve(head, path)
        if m is not None and hasattr(m, "register_forward_hook"):
            out.append((path, m))

    # CLIP image encoder
    for path in ("image_encoder", "image_encoder.model", "image_encoder.model.visual"):
        m = _resolve(head, path)
        if m is not None and hasattr(m, "register_forward_hook"):
            out.append((path, m))

    # T5 text encoder
    for path in ("text_encoder", "text_encoder.encoder"):
        m = _resolve(head, path)
        if m is not None and hasattr(m, "register_forward_hook"):
            out.append((path, m))

    # DiT (WanModel)
    dit = _resolve(head, "model")
    if dit is not None:
        if hasattr(dit, "patch_embedding"):
            out.append(("model.patch_embedding", dit.patch_embedding))
        blocks = getattr(dit, "blocks", None)
        if blocks is not None and hasattr(blocks, "__len__"):
            n = len(blocks)
            if n > 0:
                indices = sorted(set([0, n // 2, n - 1]))
                for i in indices:
                    try:
                        out.append((f"model.blocks[{i}]", blocks[i]))
                    except Exception:
                        pass
        if hasattr(dit, "head"):
            out.append(("model.head", dit.head))

    return out


# ----------------------------------------------------------------------------
# Hook installation
# ----------------------------------------------------------------------------


def install_hooks(head: Any) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Returns (records_by_name, handle_list). Hooks store first-call shapes only."""
    records: dict[str, dict[str, Any]] = {}
    handles = []

    for name, module in _candidate_modules(head):
        def make_hook(mod_name: str = name):
            def hook(_mod, inputs, outputs):
                if mod_name in records:
                    return  # only capture first call per module to avoid AR cache noise
                rec: dict[str, Any] = {"module": mod_name}
                # Inputs (just the first arg)
                in_shape, in_dtype = ([], "n/a")
                if inputs:
                    in_shape, in_dtype = _shape_of(inputs[0])
                rec["input_shape"] = in_shape
                rec["input_dtype"] = in_dtype
                # Outputs (single tensor or first of tuple)
                out_obj = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
                out_shape, out_dtype = _shape_of(out_obj)
                rec["output_shape"] = out_shape
                rec["output_dtype"] = out_dtype
                rec["module_class"] = type(_mod).__name__
                records[mod_name] = rec
            return hook
        try:
            h = module.register_forward_hook(make_hook())
            handles.append(h)
        except Exception as e:
            logger.warning("Could not hook %s: %s", name, e)

    return records, handles


# ----------------------------------------------------------------------------
# VAE decode helper (re-used in eval_baseline.py)
# ----------------------------------------------------------------------------


def decode_video_latents(head: Any, video_pred: torch.Tensor, out_path: Path) -> bool:
    try:
        import imageio
        from einops import rearrange
    except Exception:
        return False
    if not hasattr(head, "vae"):
        return False
    try:
        with torch.inference_mode():
            params = next(head.vae.parameters(), None)
            if params is None:
                return False
            vp = video_pred.to(params.device, dtype=params.dtype)
            frames = head.vae.decode(
                vp,
                tiled=getattr(head, "tiled", True),
                tile_size=(getattr(head, "tile_size_height", 34), getattr(head, "tile_size_width", 34)),
                tile_stride=(getattr(head, "tile_stride_height", 18), getattr(head, "tile_stride_width", 16)),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames = ((frames.float() + 1.0) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), list(frames), fps=5, codec="libx264")
        return True
    except Exception as e:
        logger.warning("VAE decode failed: %s", e)
        return False


# ----------------------------------------------------------------------------
# Build the audit table
# ----------------------------------------------------------------------------


def _build_records(
    hook_records: dict[str, dict[str, Any]],
    head: Any,
    video_pred: torch.Tensor | None,
    decoded_ok: bool,
) -> list[TensorRecord]:
    rows: list[TensorRecord] = []

    # Hook-derived rows
    for name, rec in hook_records.items():
        out_shape = list(rec.get("output_shape") or [])
        spatial, temporal = _classify_layout(out_shape, name)
        is_dit = "model" in name and "vae" not in name
        is_vae = "vae" in name
        is_token = "patch_embedding" in name or "block" in name
        decodable = is_vae and "decoder" not in name  # VAE encoder output -> can be VAE-decoded
        # NOTE: DiT output latents go through VAE decode at the end (see video_pred row).
        perturbable = True  # any tensor on the forward pass can be perturbed in principle
        notes = []
        if is_token:
            notes.append("token-sequence after patch embedding (B, N, D); N = T_lat * H_lat * W_lat")
        if is_vae and "encoder" in name:
            notes.append("VAE-encoded observation latents (decode with vae.decode)")
        if is_vae and "decoder" in name:
            notes.append("VAE decoder output (RGB pixels in [-1, 1])")
        if name.startswith("text_encoder"):
            notes.append("UMT5 text features (B, L, D) — semantic, not spatial")
        if name.startswith("image_encoder"):
            notes.append("CLIP image features — global, not spatial-aligned to obs")
        if is_dit and "head" in name:
            notes.append("DiT output (post denoising); reshape and VAE-decode for video")

        rows.append(TensorRecord(
            name=name,
            source=f"forward hook: {rec.get('module_class', '?')}",
            shape=out_shape,
            dtype=rec.get("output_dtype", "n/a"),
            rank=len(out_shape),
            spatial=spatial,
            temporal=temporal,
            decodable=decodable,
            perturbable=perturbable,
            notes="; ".join(notes),
        ))

    # Stored attribute rows (set by lazy_joint_video_action during forward)
    for attr in ("clip_feas", "ys", "language"):
        v = getattr(head, attr, None)
        if v is None:
            continue
        shape, dtype = _shape_of(v)
        spatial, temporal = _classify_layout(shape, attr)
        rows.append(TensorRecord(
            name=f"head.{attr}",
            source="cached on action_head after forward",
            shape=shape,
            dtype=dtype,
            rank=len(shape),
            spatial=spatial,
            temporal=temporal,
            decodable=(attr == "ys"),  # ys are VAE-encoded image features
            perturbable=True,
            notes={
                "clip_feas": "CLIP global features for the conditioning image",
                "ys": "VAE-encoded conditioning latents (decode with vae.decode)",
                "language": "T5-encoded text embedding",
            }.get(attr, ""),
        ))

    # Returned video_pred (DiT output after denoising)
    if video_pred is not None:
        shape, dtype = _shape_of(video_pred)
        rows.append(TensorRecord(
            name="video_pred (returned by lazy_joint_forward_causal)",
            source="policy.trained_model.action_head.lazy_joint_video_action -> output.transpose(1, 2)",
            shape=shape,
            dtype=dtype,
            rank=len(shape),
            spatial=True,
            temporal=True,
            decodable=decoded_ok,
            perturbable=True,
            notes=("Predicted future-video latents in DiT latent space; "
                   "decode with action_head.vae.decode(...)"),
        ))

    return rows


# ----------------------------------------------------------------------------
# Pretty writers
# ----------------------------------------------------------------------------


def _write_csv(rows: list[TensorRecord], out: Path) -> None:
    with out.open("w") as f:
        w = csv.writer(f)
        w.writerow(["name", "source", "shape", "dtype", "rank", "spatial", "temporal",
                    "decodable", "perturbable", "notes"])
        for r in rows:
            w.writerow([r.name, r.source, "x".join(str(s) for s in r.shape) or "?",
                        r.dtype, r.rank, int(r.spatial), int(r.temporal),
                        int(r.decodable), int(r.perturbable), r.notes])


def _write_md(rows: list[TensorRecord], out: Path, header: dict[str, Any]) -> None:
    lines = [
        "# E0.2 Latent Access Audit",
        "",
        f"- Checkpoint: `{header.get('checkpoint')}`",
        f"- Example: `{header.get('example_id')}`",
        f"- Embodiment: `{header.get('embodiment')}`",
        f"- Inference call: `lazy_joint_forward_causal` (causal AR head)",
        "",
        "| Tensor | Shape | Dtype | Spatial | Temporal | Decodable | Perturbable | Notes |",
        "|--------|-------|-------|:-------:|:--------:|:---------:|:-----------:|-------|",
    ]
    for r in rows:
        shape_str = "×".join(str(s) for s in r.shape) if r.shape else "?"
        lines.append(
            f"| `{r.name}` | {shape_str} | {r.dtype} | "
            f"{'✓' if r.spatial else '·'} | {'✓' if r.temporal else '·'} | "
            f"{'✓' if r.decodable else '·'} | {'✓' if r.perturbable else '·'} | "
            f"{r.notes} |"
        )
    out.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def run_audit(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.task_suite).resolve()
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    if not manifest:
        raise RuntimeError(f"Empty manifest at {manifest_path}")

    if args.example_id is not None:
        candidates = [e for e in manifest if e.get("example_id") == args.example_id]
        if not candidates:
            raise KeyError(f"example_id={args.example_id} not in manifest")
        entry = candidates[0]
    else:
        idx = max(0, min(args.example_index, len(manifest) - 1))
        entry = manifest[idx]
    logger.info("Auditing example: %s (frame=%d)", entry["example_id"], entry["frame_index"])

    init_dist_single_process()

    logger.info("Loading model from %s ...", args.checkpoint)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        model_path=args.checkpoint,
        device=args.device,
    )
    head = policy.trained_model.action_head
    logger.info("Model loaded. Installing hooks ...")
    hook_records, handles = install_hooks(head)
    logger.info("Hooked %d modules.", len(handles))

    obs = build_obs(entry)
    reset_causal_state(policy)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    with torch.inference_mode():
        result_batch, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    if torch.is_tensor(video_pred):
        video_pred_cpu = video_pred.detach().to("cpu")
    else:
        video_pred_cpu = None

    # Decode video_pred if requested for sanity
    decoded_ok = False
    if video_pred_cpu is not None:
        decoded_ok = decode_video_latents(
            head, video_pred, out_dir / "decoded_video_pred.mp4"
        )

    rows = _build_records(hook_records, head, video_pred_cpu, decoded_ok)

    header = {
        "checkpoint": args.checkpoint,
        "example_id": entry["example_id"],
        "embodiment": EMBODIMENT_TAG,
        "instruction": entry.get("instruction"),
        "task_group": entry.get("task_group"),
        "frame_index": entry.get("frame_index"),
        "video_pred_decoded": decoded_ok,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    payload = {"header": header, "tensors": [r.asdict() for r in rows]}
    (out_dir / "latent_audit.json").write_text(json.dumps(payload, indent=2))
    _write_csv(rows, out_dir / "latent_audit.csv")
    _write_md(rows, out_dir / "latent_audit.md", header)

    logger.info("Wrote %d tensor rows to %s", len(rows), out_dir)
    if decoded_ok:
        logger.info("Decoded video saved -> %s", out_dir / "decoded_video_pred.mp4")
    else:
        logger.warning("Skipped video decode (no VAE / decode failed).")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", required=True, help="manifest.json from build_task_suite.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--example_index", type=int, default=0,
                   help="Manifest entry to audit (ignored if --example_id is given).")
    p.add_argument("--example_id", default=None,
                   help="Specific example_id from the manifest to audit.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    try:
        run_audit(args)
    except Exception as e:
        logger.error("latent_audit failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
