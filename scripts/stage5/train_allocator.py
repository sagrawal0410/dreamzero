#!/usr/bin/env python3
"""Stage 5 / Train — Distil Stage-1 perturbation maps into a lightweight allocator.

Implements **E5.1 distillation** training. Also covers:

    * E5.3 generalization splits — pass `--val_groups C_distractor,D_global_context`
      (or any subset of task-group names) and the train/val split is by group
      instead of random. Use this to test allocator generalization to unseen
      task families.
    * E5.4 input ablations — pass `--ablation` to launch four sequential runs
      with the standard input combinations, each writing its own subfolder and
      checkpoint:
        vision_only, vision_lang, vision_proprio, vision_lang_proprio_history.

Dataset
-------
`PerturbMapDataset` (see `allocator_model.py`) reads:

    runs/stage1_maps/<example_id>/heatmaps.npz   ← target action-causal map
    runs/stage0_suite/manifest.json              ← image path, proprio, instruction,
                                                   episode → action history

The target heatmap is L1-normalised to a probability simplex; the loss is
KL-divergence + a Spearman-style soft-rank term to encourage correct
ordering. Standard MSE is also reported for compatibility with paper
appendix tables.

Example (single run):

    python scripts/stage5/train_allocator.py \\
        --maps_dir runs/stage1_maps \\
        --task_suite runs/stage0_suite/manifest.json \\
        --epochs 60 --batch_size 16 --lr 1e-3 \\
        --use_lang --use_proprio --use_history \\
        --output_dir runs/stage5_allocator/full

Example (E5.4 input ablations, one shell call, four runs):

    python scripts/stage5/train_allocator.py \\
        --maps_dir runs/stage1_maps \\
        --task_suite runs/stage0_suite/manifest.json \\
        --epochs 60 --batch_size 16 \\
        --ablation \\
        --output_dir runs/stage5_allocator_ablation

Example (E5.3 generalization, hold out distractor + global-context tasks):

    python scripts/stage5/train_allocator.py \\
        --maps_dir runs/stage1_maps \\
        --task_suite runs/stage0_suite/manifest.json \\
        --val_groups C_distractor,D_global_context \\
        --epochs 60 --batch_size 16 \\
        --output_dir runs/stage5_allocator_generalization

Outputs (per run directory):

    config.json                — hyperparameters and the input-ablation flags.
    train_log.csv              — per-epoch train/val loss + map metrics.
    train_log.json             — same content, JSON.
    allocator.pt               — best-val checkpoint (state_dict + config).
    predicted_maps/<id>/heatmaps.npz   — predicted maps in Stage-1 layout
                                        for every val example (and train if
                                        --dump_train_predictions is set).
    plots/plot_loss_curve.png  - training-loss curve.
    plots/plot_val_metrics.png - val metric curves over epochs.
    summary.json               — final-epoch + best-epoch metrics.
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage5.train")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from allocator_model import (  # noqa: E402
    Allocator, PerturbMapDataset, all_metrics,
    save_predictions_npz, _normalize_target,
)
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
sys.path.insert(0, str(STAGE0_DIR))
from _common import configure_neurips_matplotlib  # noqa: E402


# ============================================================================
# Loss
# ============================================================================


def _kl_div(pred_logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    """KL(target || softmax(pred)) per example, averaged over the batch.

    pred_logits : (B, H, W) raw scores
    target_prob : (B, H, W) non-negative, sums to 1 per example (or all-zero).
    """
    B = pred_logits.shape[0]
    flat_logits = pred_logits.view(B, -1)
    flat_target = target_prob.view(B, -1)
    log_q = F.log_softmax(flat_logits, dim=-1)
    eps = 1e-8
    p = flat_target.clamp_min(0.0)
    p_sum = p.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p / p_sum
    kl = (p * (torch.log(p + eps) - log_q)).sum(dim=-1)
    return kl.mean()


def _soft_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson(pred_flat, target_flat) per example, batch-mean."""
    B = pred.shape[0]
    x = pred.view(B, -1).float()
    y = target.view(B, -1).float()
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    xn = x / (x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    yn = y / (y.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    cos = (xn * yn).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()


def composite_loss(pred_logits: torch.Tensor, target_prob: torch.Tensor,
                   w_kl: float = 1.0, w_rank: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    kl = _kl_div(pred_logits, target_prob)
    rank = _soft_rank_loss(pred_logits, target_prob)
    total = w_kl * kl + w_rank * rank
    return total, {"kl": float(kl.item()), "rank": float(rank.item()),
                   "total": float(total.item())}


# ============================================================================
# Eval
# ============================================================================


@torch.no_grad()
def evaluate(model: Allocator, loader: DataLoader, device: str,
             k_pcts: list[float]) -> dict[str, Any]:
    model.eval()
    losses = []
    metrics_acc: dict[str, list[float]] = {}
    pred_lat_ms: list[float] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        lang = batch["lang_emb"].to(device, non_blocking=True)
        history = batch["history"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        t0 = time.perf_counter()
        pred_logits = model(image, proprio, lang, history)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        pred_lat_ms.append((time.perf_counter() - t0) * 1000.0 / max(1, image.shape[0]))
        loss, _logs = composite_loss(pred_logits, target)
        losses.append(float(loss.item()))
        # Convert to probability for metric computation (softmax over flattened)
        B, H, W = pred_logits.shape
        prob = F.softmax(pred_logits.view(B, -1), dim=-1).view(B, H, W).cpu().numpy()
        tgt = target.cpu().numpy()
        for b in range(B):
            m = all_metrics(prob[b], tgt[b], k_pcts)
            for k, v in m.items():
                metrics_acc.setdefault(k, []).append(v)
    out: dict[str, Any] = {"val_loss": float(np.mean(losses)) if losses else float("nan"),
                           "val_pred_latency_ms": float(np.mean(pred_lat_ms)) if pred_lat_ms else float("nan")}
    for k, vs in metrics_acc.items():
        arr = np.array([v for v in vs if v is not None and not (isinstance(v, float) and math.isnan(v))])
        out[f"val_{k}_mean"] = float(arr.mean()) if arr.size else float("nan")
        out[f"val_{k}_median"] = float(np.median(arr)) if arr.size else float("nan")
    model.train()
    return out


def dump_predictions(model: Allocator, loader: DataLoader, device: str,
                     out_dir: Path, primary_op: str, primary_metric: str,
                     target_T: int = 1) -> int:
    """Save softmax-normalised predictions in Stage-1 npz layout. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            lang = batch["lang_emb"].to(device, non_blocking=True)
            history = batch["history"].to(device, non_blocking=True)
            pred_logits = model(image, proprio, lang, history)
            B, H, W = pred_logits.shape
            prob = F.softmax(pred_logits.view(B, -1), dim=-1).view(B, H, W).cpu().numpy()
            ids = batch["example_id"]
            for b in range(B):
                ex_id = ids[b] if isinstance(ids, list) else (ids[b] if not torch.is_tensor(ids) else str(ids[b].item()))
                save_predictions_npz(out_dir, str(ex_id), prob[b],
                                     primary_op, primary_metric, expand_T=target_T)
                n += 1
    model.train()
    return n


# ============================================================================
# Training driver (single config)
# ============================================================================


def train_one_config(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    device = cfg["device"]
    rng_seed = int(cfg["seed"])
    torch.manual_seed(rng_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(rng_seed)

    # Dataset
    dataset = PerturbMapDataset(
        maps_dir=Path(cfg["maps_dir"]),
        manifest_path=Path(cfg["task_suite"]),
        primary_op=cfg["primary_operator"],
        primary_metric=cfg["primary_metric"],
        image_size=tuple(cfg["image_size"]),
        history_len=int(cfg["history_len"]),
        action_dim=int(cfg["action_dim"]),
        lang_dim=int(cfg["lang_dim"]),
        camera=cfg["camera"],
        task_groups=cfg.get("train_groups"),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No usable examples in {cfg['maps_dir']} matching the manifest.")

    val_groups = cfg.get("val_groups") or None
    train_set, val_set = dataset.split(
        val_groups=val_groups,
        val_fraction=float(cfg["val_fraction"]),
        seed=rng_seed,
    )
    logger.info("Dataset: %d train, %d val   target_shape=%s",
                len(train_set), len(val_set), train_set.target_shape)

    if len(train_set) == 0 or len(val_set) == 0:
        logger.warning("Empty train or val split; relaxing val_groups.")
        train_set, val_set = dataset.split(val_groups=None,
                                           val_fraction=cfg["val_fraction"],
                                           seed=rng_seed)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], drop_last=False)

    # Model
    target_shape = train_set.target_shape if train_set.target_shape != (0, 0) else val_set.target_shape
    model = Allocator(
        image_size=tuple(cfg["image_size"]),
        proprio_dim=int(train_set.proprio_dim or cfg["proprio_dim"]),
        lang_emb_dim=int(cfg["lang_dim"]),
        history_len=int(cfg["history_len"]),
        action_dim=int(cfg["action_dim"]),
        target_size=tuple(target_shape if target_shape != (0, 0) else cfg["target_size"]),
        hidden=int(cfg["hidden"]),
        use_lang=bool(cfg["use_lang"]),
        use_proprio=bool(cfg["use_proprio"]),
        use_history=bool(cfg["use_history"]),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Allocator parameters: %.2fM   target=%s   conds=%d",
                n_params / 1e6, model.target_size, model._cond_in)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["epochs"]))

    k_pcts = list(cfg["top_k_pcts"])
    log_rows: list[dict[str, Any]] = []
    best_val_pearson = -math.inf
    best_path = out_dir / "allocator.pt"
    primary_metric_name = "val_pearson_mean"

    epoch_t0 = time.perf_counter()
    for epoch in range(int(cfg["epochs"])):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            lang = batch["lang_emb"].to(device, non_blocking=True)
            history = batch["history"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            opt.zero_grad()
            pred_logits = model(image, proprio, lang, history)
            loss, _logs = composite_loss(pred_logits, target,
                                          w_kl=float(cfg["w_kl"]),
                                          w_rank=float(cfg["w_rank"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            train_losses.append(float(loss.item()))
        sched.step()

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_metrics = evaluate(model, val_loader, device, k_pcts)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(opt.param_groups[0]["lr"]),
            **val_metrics,
        }
        log_rows.append(row)
        logger.info(
            "epoch %3d  train_loss=%.4f  val_loss=%.4f  Pearson=%.3f  Spearman=%.3f  "
            "IoU@10=%.3f  pred_lat=%.2fms",
            epoch, train_loss, row.get("val_loss", float("nan")),
            row.get("val_pearson_mean", float("nan")),
            row.get("val_spearman_mean", float("nan")),
            row.get("val_iou_top_10_mean", float("nan")),
            row.get("val_pred_latency_ms", float("nan")),
        )

        # Best checkpoint
        cur = row.get(primary_metric_name)
        if cur is not None and not (isinstance(cur, float) and math.isnan(cur)) and cur > best_val_pearson:
            best_val_pearson = float(cur)
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "metrics": row}, best_path)

    elapsed = time.perf_counter() - epoch_t0
    # Persist log
    with (out_dir / "train_log.csv").open("w") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            for r in log_rows:
                w.writerow(r)
    (out_dir / "train_log.json").write_text(json.dumps(log_rows, indent=2))

    # Reload best for final eval / dumps
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Reloaded best checkpoint @ epoch %d (%s=%.3f)",
                    ckpt.get("epoch", -1), primary_metric_name,
                    ckpt.get("metrics", {}).get(primary_metric_name, float("nan")))

    # Dump predictions for val (always) + train (optional)
    dump_dir = out_dir / "predicted_maps"
    n_val = dump_predictions(model, val_loader, device, dump_dir,
                              cfg["primary_operator"], cfg["primary_metric"],
                              target_T=int(cfg.get("expand_T", 1)))
    if cfg.get("dump_train_predictions"):
        n_tr = dump_predictions(model, train_loader, device, dump_dir,
                                cfg["primary_operator"], cfg["primary_metric"],
                                target_T=int(cfg.get("expand_T", 1)))
    else:
        n_tr = 0
    logger.info("Dumped predictions: %d val + %d train -> %s", n_val, n_tr, dump_dir)

    # Final summary + plots
    final = log_rows[-1] if log_rows else {}
    best_row = max(log_rows, key=lambda r: (r.get(primary_metric_name) or -math.inf)) if log_rows else {}
    summary = {
        "n_train": len(train_set), "n_val": len(val_set),
        "elapsed_s": elapsed,
        "n_parameters": n_params,
        "final": final,
        "best": best_row,
        "config": cfg,
        "predicted_maps_dir": str(dump_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_training_curves(log_rows, out_dir / "plots")
    return summary


def _plot_training_curves(log_rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not log_rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_neurips_matplotlib()
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in log_rows]
    train_l = [r.get("train_loss", float("nan")) for r in log_rows]
    val_l = [r.get("val_loss", float("nan")) for r in log_rows]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(epochs, train_l, label="train", color="#3b75af")
    ax.plot(epochs, val_l, label="val", color="#c14b4b")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (KL + α·rank)")
    ax.set_title("Stage 5 — training loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(out_dir / f"plot_loss_curve{ext}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for k, color in [("val_pearson_mean", "#3b75af"),
                     ("val_spearman_mean", "#3b8b58"),
                     ("val_iou_top_10_mean", "#c14b4b"),
                     ("val_recall_top_10_mean", "#7a4dab")]:
        ys = [r.get(k, float("nan")) for r in log_rows]
        ax.plot(epochs, ys, label=k.replace("val_", "").replace("_mean", ""),
                color=color)
    ax.set_xlabel("epoch"); ax.set_ylabel("metric")
    ax.set_title("Stage 5 — validation map metrics")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(out_dir / f"plot_val_metrics{ext}")
    plt.close(fig)


# ============================================================================
# Ablation orchestrator (E5.4)
# ============================================================================


ABLATION_VARIANTS = {
    "vision_only":             {"use_lang": False, "use_proprio": False, "use_history": False},
    "vision_lang":             {"use_lang": True,  "use_proprio": False, "use_history": False},
    "vision_proprio":          {"use_lang": False, "use_proprio": True,  "use_history": False},
    "vision_lang_proprio_history":
                               {"use_lang": True,  "use_proprio": True,  "use_history": True},
}


def run_ablation(base_cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, overrides in ABLATION_VARIANTS.items():
        cfg = dict(base_cfg)
        cfg.update(overrides)
        cfg["ablation_variant"] = name
        sub_dir = out_root / name
        logger.info("=== Ablation run: %s ===  (use_lang=%s use_proprio=%s use_history=%s)",
                    name, cfg["use_lang"], cfg["use_proprio"], cfg["use_history"])
        try:
            summaries[name] = train_one_config(cfg, sub_dir)
        except Exception as e:
            logger.error("Ablation %s failed: %s", name, e, exc_info=True)
            summaries[name] = {"error": str(e)}
    (out_root / "ablation_summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description=__doc__)
    p.add_argument("--maps_dir", required=True,
                   help="Directory written by Stage-1 causal_maps_compute.py.")
    p.add_argument("--task_suite", required=True,
                   help="manifest.json from Stage 0 build_task_suite.py.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    # Optimisation
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--w_kl", type=float, default=1.0)
    p.add_argument("--w_rank", type=float, default=0.5)
    # Splits
    p.add_argument("--val_fraction", type=float, default=0.2,
                   help="Random val fraction when --val_groups is empty.")
    p.add_argument("--train_groups", default="",
                   help="Comma-separated task groups to KEEP for training "
                        "(default: all).")
    p.add_argument("--val_groups", default="",
                   help="Comma-separated task groups for the val split. If set, "
                        "examples in these groups are held out for validation "
                        "(E5.3 generalization).")
    # Inputs / architecture
    p.add_argument("--image_height", type=int, default=180)
    p.add_argument("--image_width", type=int, default=320)
    p.add_argument("--target_h", type=int, default=11)
    p.add_argument("--target_w", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--proprio_dim", type=int, default=8)
    p.add_argument("--lang_dim", type=int, default=64)
    p.add_argument("--history_len", type=int, default=4)
    p.add_argument("--action_dim", type=int, default=8)
    p.add_argument("--camera", default="video.exterior_image_1_left")
    p.add_argument("--use_lang", action="store_true", default=True)
    p.add_argument("--no_lang", dest="use_lang", action="store_false")
    p.add_argument("--use_proprio", action="store_true", default=True)
    p.add_argument("--no_proprio", dest="use_proprio", action="store_false")
    p.add_argument("--use_history", action="store_true", default=True)
    p.add_argument("--no_history", dest="use_history", action="store_false")
    # Stage-1 source
    p.add_argument("--primary_operator", default="local_mean")
    p.add_argument("--primary_metric", default="action_l2")
    p.add_argument("--top_k_pcts", default="5,10,20",
                   help="k% for IoU/recall metrics during eval.")
    p.add_argument("--expand_T", type=int, default=1,
                   help="When dumping predictions in Stage-1 layout, broadcast "
                        "the 2-D map to this many time slices.")
    p.add_argument("--dump_train_predictions", action="store_true",
                   help="Also save predicted maps for the train set.")
    p.add_argument("--ablation", action="store_true",
                   help="Launch four sequential runs covering the E5.4 input ablations.")
    args = p.parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg: dict[str, Any] = {
        "maps_dir": str(Path(args.maps_dir).resolve()),
        "task_suite": str(Path(args.task_suite).resolve()),
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "w_kl": args.w_kl,
        "w_rank": args.w_rank,
        "val_fraction": args.val_fraction,
        "train_groups": [g for g in args.train_groups.split(",") if g.strip()] or None,
        "val_groups": [g for g in args.val_groups.split(",") if g.strip()] or None,
        "image_size": [args.image_height, args.image_width],
        "target_size": [args.target_h, args.target_w],
        "hidden": args.hidden,
        "proprio_dim": args.proprio_dim,
        "lang_dim": args.lang_dim,
        "history_len": args.history_len,
        "action_dim": args.action_dim,
        "camera": args.camera,
        "use_lang": args.use_lang,
        "use_proprio": args.use_proprio,
        "use_history": args.use_history,
        "primary_operator": args.primary_operator,
        "primary_metric": args.primary_metric,
        "top_k_pcts": [float(x) for x in args.top_k_pcts.split(",") if x.strip()],
        "expand_T": args.expand_T,
        "dump_train_predictions": args.dump_train_predictions,
    }

    if args.ablation:
        run_ablation(base_cfg, out_root)
    else:
        train_one_config(base_cfg, out_root)


if __name__ == "__main__":
    main()
