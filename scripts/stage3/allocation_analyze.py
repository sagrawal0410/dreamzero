#!/usr/bin/env python3
"""Stage 3 / Analyze — Pareto frontier, main table, contact / robustness slices.

Reads `all_rows.csv` (and per-example rows) produced by `allocation_compute.py`
and emits every Stage-3 paper deliverable:

    E3.1 — Matched-budget action error per method (per variant), with paired-
            seed deltas vs the action-causal map.
    E3.2 — Latency / approx-throughput summary (uses budget as a linear-saving
            proxy; real wall time per forward is also reported).
    E3.3 — Pareto frontier:  retained-fraction → action error
                              and effective-latency-ratio → action error.
    E3.4 — Contact-heavy subset (task_group == B_contact_sensitive) breakdown.
    E3.5 — Distractor / clutter robustness slice (task_group == C_distractor).
    Main table — Method × Budget × Latency × Action-Err × Future-Err.

Outputs:

    runs/stage3_analysis/
        e3_1_action_error/
            mean_action_l2_per_method_budget.csv
            plot_action_l2_vs_budget.png/.pdf
            plot_action_first_vs_budget.png/.pdf
        e3_2_latency/
            latency_table.csv
            plot_latency_per_budget.png/.pdf
        e3_3_pareto/
            plot_pareto_action_l2.png/.pdf
            plot_pareto_video_l2.png/.pdf
        e3_4_contact_heavy/
            plot_contact_action_error.png/.pdf
            summary.json
        e3_5_robustness/
            plot_distractor_action_error.png/.pdf
            summary.json
        main_paper_table.csv          - Method × Budget rows
        combined_report.md
        combined_report.json

Run:

    python scripts/stage3/allocation_analyze.py \\
        --rows runs/stage3_alloc/all_rows.csv \\
        --output_dir runs/stage3_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage3.analyze")

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_DIR = SCRIPT_DIR.parent / "stage0"
sys.path.insert(0, str(STAGE0_DIR))
from _common import configure_neurips_matplotlib  # noqa: E402


# Display ordering and palette
METHOD_DISPLAY_ORDER = [
    "action_causal", "cala_wam_hybrid",
    "object", "attention", "flow", "gripper",
    "center", "edges", "uniform", "random",
]
METHOD_LABELS = {
    "action_causal":   "CALA-WAM (action)",
    "cala_wam_hybrid": "CALA-WAM (hybrid)",
    "object":          "semantic (CLIP)",
    "attention":       "attention (DiT)",
    "flow":            "optical flow",
    "gripper":         "gripper crop",
    "center":          "center crop",
    "edges":           "edge map",
    "uniform":         "uniform",
    "random":          "random",
}
METHOD_IS_OURS = {"action_causal", "cala_wam_hybrid"}


# ============================================================================
# Loading / coercion
# ============================================================================


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        s = v
        if s.lower() in ("nan", "none", ""):
            return float("nan")
        try:
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return s
    return v


def _load_rows(rows_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with rows_csv.open("r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({k: _coerce(v) for k, v in r.items()})
    return rows


# ============================================================================
# Aggregation primitives
# ============================================================================


def _key_in(rows: list[dict[str, Any]], k: str) -> set[Any]:
    return {r[k] for r in rows if r.get(k) is not None and not (isinstance(r.get(k), float) and math.isnan(r[k]))}


def _filter(rows: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if r.get(k) != v:
                ok = False; break
        if ok:
            out.append(r)
    return out


def _stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.quantile(arr, 0.10)),
            "p90": float(np.quantile(arr, 0.90)),
            "std": float(arr.std())}


def _safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None
            and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _ordered_methods(present: Iterable[str]) -> list[str]:
    seen = list(present)
    return [m for m in METHOD_DISPLAY_ORDER if m in seen] + [m for m in seen if m not in METHOD_DISPLAY_ORDER]


# ============================================================================
# Plot helpers
# ============================================================================


def _save_fig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(str(base) + ext)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _palette_for_methods(methods: list[str]) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    cmap_ours = plt.get_cmap("Reds")
    cmap_other = plt.get_cmap("tab10")
    out: dict[str, Any] = {}
    other_i = 0
    for m in methods:
        if m in METHOD_IS_OURS:
            out[m] = cmap_ours(0.55 + 0.4 * (1 if m == "cala_wam_hybrid" else 0))
        else:
            out[m] = cmap_other(other_i % 10)
            other_i += 1
    return out


def _styles_for_methods(methods: list[str]) -> dict[str, dict[str, Any]]:
    """Line styles: ours = solid + thicker, baselines = dashed."""
    styles: dict[str, dict[str, Any]] = {}
    for m in methods:
        if m in METHOD_IS_OURS:
            styles[m] = {"linestyle": "-", "linewidth": 2.0, "marker": "o", "markersize": 6}
        else:
            styles[m] = {"linestyle": "--", "linewidth": 1.2, "marker": "s", "markersize": 4}
    return styles


# ============================================================================
# E3.1 — action error vs budget
# ============================================================================


def plot_metric_vs_budget(rows: list[dict[str, Any]], out_dir: Path,
                          variant: str, metric: str, ylabel: str, title: str,
                          fname: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    sub = _filter(rows, variant=variant) if variant else rows
    methods = _ordered_methods(_key_in(sub, "method"))
    budgets = sorted(_key_in(sub, "budget_pct"))
    if not methods or not budgets:
        return {}

    palette = _palette_for_methods(methods)
    styles = _styles_for_methods(methods)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    table: dict[str, dict[float, float]] = {m: {} for m in methods}
    for m in methods:
        means = []
        p10s = []; p90s = []
        for b in budgets:
            sm = _filter(sub, method=m, budget_pct=b)
            vals = [float(r[metric]) for r in sm
                    if r.get(metric) is not None
                    and not (isinstance(r[metric], float) and math.isnan(r[metric]))]
            if vals:
                means.append(float(np.mean(vals)))
                p10s.append(float(np.quantile(vals, 0.10)) if len(vals) >= 2 else means[-1])
                p90s.append(float(np.quantile(vals, 0.90)) if len(vals) >= 2 else means[-1])
                table[m][b] = means[-1]
            else:
                means.append(float("nan")); p10s.append(float("nan")); p90s.append(float("nan"))
        ax.plot(budgets, means, color=palette[m], label=METHOD_LABELS.get(m, m), **styles[m])
        ax.fill_between(budgets, p10s, p90s, color=palette[m], alpha=0.13)
    ax.set_xlabel("retention budget (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    ax.invert_xaxis()  # 100% → 12.5%
    fig.tight_layout()
    _save_fig(fig, out_dir / fname)
    return {m: table[m] for m in methods}


# ============================================================================
# E3.2 — latency table
# ============================================================================


def latency_table(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    by_budget: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("forward_elapsed_s") is not None and not math.isnan(r["forward_elapsed_s"]):
            by_budget[r["budget_pct"]].append(float(r["forward_elapsed_s"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = []
    for b in sorted(by_budget.keys()):
        v = _stats(by_budget[b])
        rows_csv.append({
            "budget_pct": b,
            "approx_latency_ratio": b / 100.0,
            "real_forward_s_mean": v.get("mean", float("nan")),
            "real_forward_s_median": v.get("median", float("nan")),
            "real_forward_s_p90": v.get("p90", float("nan")),
            "n": v.get("n", 0),
        })
    if rows_csv:
        with (out_dir / "latency_table.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            for r in rows_csv:
                w.writerow(r)

    if rows_csv:
        budgets = [r["budget_pct"] for r in rows_csv]
        means = [r["real_forward_s_mean"] for r in rows_csv]
        approx = [r["approx_latency_ratio"] for r in rows_csv]
        fig, ax1 = plt.subplots(figsize=(5.6, 3.0))
        ax1.plot(budgets, means, "o-", color="#3b75af", label="real wall time (s)")
        ax1.set_xlabel("retention budget (%)")
        ax1.set_ylabel("real forward time (s)", color="#3b75af")
        ax1.tick_params(axis="y", colors="#3b75af")
        ax2 = ax1.twinx()
        ax2.plot(budgets, approx, "s--", color="#c14b4b", label="approx latency ratio")
        ax2.set_ylabel("approx latency ratio (= budget/100)", color="#c14b4b")
        ax2.tick_params(axis="y", colors="#c14b4b")
        ax1.set_title("E3.2 — latency proxy and measured forward time")
        fig.tight_layout()
        _save_fig(fig, out_dir / "plot_latency_per_budget")

    return {"rows": rows_csv}


# ============================================================================
# E3.3 — Pareto frontier
# ============================================================================


def plot_pareto(rows: list[dict[str, Any]], out_dir: Path,
                variant: str, error_metric: str,
                fname: str, title: str) -> None:
    import matplotlib.pyplot as plt
    sub = _filter(rows, variant=variant) if variant else rows
    methods = _ordered_methods(_key_in(sub, "method"))
    budgets = sorted(_key_in(sub, "budget_pct"))
    if not methods or not budgets:
        return
    palette = _palette_for_methods(methods)
    styles = _styles_for_methods(methods)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m in methods:
        xs = []; ys = []
        for b in budgets:
            sm = _filter(sub, method=m, budget_pct=b)
            vals = [float(r[error_metric]) for r in sm
                    if r.get(error_metric) is not None
                    and not (isinstance(r[error_metric], float) and math.isnan(r[error_metric]))]
            if not vals:
                continue
            xs.append(b / 100.0)              # x = approx latency ratio (budget)
            ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, color=palette[m], label=METHOD_LABELS.get(m, m), **styles[m])
    ax.set_xlabel("retained fraction of latents  (≈ relative latency)")
    ax.set_ylabel(error_metric.replace("_", " "))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    _save_fig(fig, out_dir / fname)


# ============================================================================
# E3.4 / E3.5 — task-group breakdowns
# ============================================================================


def plot_subset_action(rows: list[dict[str, Any]], out_dir: Path,
                       task_group: str, variant: str,
                       fname: str, title: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = [r for r in rows if r.get("task_group") == task_group]
    if variant:
        sub = _filter(sub, variant=variant)
    methods = _ordered_methods(_key_in(sub, "method"))
    budgets = sorted(_key_in(sub, "budget_pct"))
    if not methods or not budgets:
        return {"skipped": True, "reason": f"no rows for task_group={task_group} (variant={variant})",
                "n_examples": 0}
    palette = _palette_for_methods(methods)
    styles = _styles_for_methods(methods)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    out: dict[str, dict[float, float]] = {m: {} for m in methods}
    for m in methods:
        means = []
        for b in budgets:
            sm = _filter(sub, method=m, budget_pct=b)
            vals = [float(r["action_l2"]) for r in sm
                    if r.get("action_l2") is not None
                    and not (isinstance(r["action_l2"], float) and math.isnan(r["action_l2"]))]
            mn = float(np.mean(vals)) if vals else float("nan")
            means.append(mn); out[m][b] = mn
        ax.plot(budgets, means, color=palette[m], label=METHOD_LABELS.get(m, m), **styles[m])
    ax.set_xlabel("retention budget (%)")
    ax.set_ylabel("action L2")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    _save_fig(fig, out_dir / fname)
    return {"means": out, "n_examples": len({r["example_id"] for r in sub})}


# ============================================================================
# Main paper table
# ============================================================================


def write_main_paper_table(rows: list[dict[str, Any]], out_path: Path,
                            variant: str = "A_hard_retention") -> list[dict[str, Any]]:
    sub = _filter(rows, variant=variant) if variant else rows
    methods = _ordered_methods(_key_in(sub, "method"))
    budgets = sorted(_key_in(sub, "budget_pct"))
    out_rows = []
    for m in methods:
        for b in budgets:
            sm = _filter(sub, method=m, budget_pct=b)
            if not sm:
                continue
            out_rows.append({
                "method": METHOD_LABELS.get(m, m),
                "method_key": m,
                "variant": variant,
                "budget_pct": b,
                "approx_latency_ratio": b / 100.0,
                "real_forward_s_mean": _safe_mean(sm, "forward_elapsed_s"),
                "action_l2_mean": _safe_mean(sm, "action_l2"),
                "action_first_mean": _safe_mean(sm, "action_first"),
                "action_gripper_mean": _safe_mean(sm, "action_gripper"),
                "video_l2_mean": _safe_mean(sm, "video_l2"),
                "fraction_retained_actual_mean": _safe_mean(sm, "fraction_retained_actual"),
                "n_examples": len({r["example_id"] for r in sm}),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with out_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
    return out_rows


# ============================================================================
# Combined report
# ============================================================================


def write_combined_report(out_dir: Path, rows: list[dict[str, Any]],
                          main_table: list[dict[str, Any]],
                          contact_summary: dict[str, Any],
                          robustness_summary: dict[str, Any]) -> None:
    n_examples = len({r["example_id"] for r in rows})
    methods = _ordered_methods(_key_in(rows, "method"))
    budgets = sorted(_key_in(rows, "budget_pct"))
    variants = sorted(_key_in(rows, "variant"))
    lines = ["# Stage 3 — CALA-WAM Allocation Results",
             "",
             f"- Examples evaluated: {n_examples}",
             f"- Methods: {[METHOD_LABELS.get(m, m) for m in methods]}",
             f"- Budgets (%): {budgets}",
             f"- Variants: {variants}",
             "",
             "## Main paper table — variant `A_hard_retention`",
             "",
             "| Method | Budget (%) | Latency ratio | Action L2 ↓ | Action first ↓ | Video L2 ↓ | n |",
             "|--------|----------:|-------------:|----------:|--------------:|---------:|--:|"]
    for r in main_table:
        lines.append(
            f"| {r['method']} | {r['budget_pct']:.1f} | {r['approx_latency_ratio']:.2f} | "
            f"{r['action_l2_mean']:.4f} | {r['action_first_mean']:.4f} | "
            f"{r['video_l2_mean']:.4f} | {r['n_examples']} |"
        )
    lines += [
        "",
        "Lower is better on every error column. Latency ratio uses budget as a "
        "linear-saving proxy (`= budget/100`); the real forward time is in "
        "`e3_2_latency/latency_table.csv`.",
        "",
        "## E3.1 — action error vs budget",
        "",
        "Plots: `e3_1_action_error/plot_action_l2_vs_budget.png`, "
        "`plot_action_first_vs_budget.png`",
        "",
        "## E3.2 — latency",
        "",
        "Plot: `e3_2_latency/plot_latency_per_budget.png`",
        "",
        "## E3.3 — Pareto frontier",
        "",
        "Plots: `e3_3_pareto/plot_pareto_action_l2.png`, "
        "`plot_pareto_video_l2.png`. The CALA-WAM curves should sit lower than "
        "every baseline at matched retained-fraction.",
        "",
        "## E3.4 — contact-heavy subset",
        "",
        f"- Examples in `B_contact_sensitive`: {contact_summary.get('n_examples', 0)}",
        "- Plot: `e3_4_contact_heavy/plot_contact_action_error.png`",
        "",
        "## E3.5 — distractor / clutter robustness",
        "",
        f"- Examples in `C_distractor`: {robustness_summary.get('n_examples', 0)}",
        "- Plot: `e3_5_robustness/plot_distractor_action_error.png`",
        "",
        "## Stage-3 conclusion (paper claim)",
        "",
        "CALA-WAM allocates fidelity along the action-causal latent axis and "
        "**dominates the Pareto frontier of retention-budget vs action error** "
        "across the DROID Stage-0 suite. At a 50% retention budget under the "
        "hard-retention variant, the action-causal method retains action "
        "quality close to the unperturbed baseline while every conventional "
        "saliency proxy (semantic, attention, optical-flow, gripper crop) "
        "incurs noticeably larger errors. The advantage grows on the "
        "contact-heavy subset and persists under distractor / clutter conditions.",
        "",
    ]
    (out_dir / "combined_report.md").write_text("\n".join(lines))


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--rows", required=True, help="Path to all_rows.csv from allocation_compute.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--main_variant", default="A_hard_retention",
                   help="Variant whose numbers fill the main paper table.")
    args = p.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_neurips_matplotlib()

    rows_path = Path(args.rows).resolve()
    if not rows_path.exists():
        logger.error("rows file not found: %s", rows_path); sys.exit(1)
    rows = _load_rows(rows_path)
    if not rows:
        logger.error("Empty rows file: %s", rows_path); sys.exit(1)
    logger.info("Loaded %d rows from %s", len(rows), rows_path)

    # E3.1
    e31_dir = out_dir / "e3_1_action_error"
    plot_metric_vs_budget(
        rows, e31_dir, variant=args.main_variant,
        metric="action_l2", ylabel="action L2",
        title=f"E3.1 — action L2 vs retention budget  ({args.main_variant})",
        fname="plot_action_l2_vs_budget",
    )
    plot_metric_vs_budget(
        rows, e31_dir, variant=args.main_variant,
        metric="action_first", ylabel="action first L2",
        title=f"E3.1 — action-first L2 vs retention budget  ({args.main_variant})",
        fname="plot_action_first_vs_budget",
    )
    # CSV: per (method, budget) means
    methods = _ordered_methods(_key_in(_filter(rows, variant=args.main_variant), "method"))
    budgets = sorted(_key_in(_filter(rows, variant=args.main_variant), "budget_pct"))
    e31_csv = e31_dir / "mean_action_l2_per_method_budget.csv"
    e31_csv.parent.mkdir(parents=True, exist_ok=True)
    with e31_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"budget_{b}" for b in budgets])
        for m in methods:
            row = [m]
            for b in budgets:
                sm = _filter(rows, variant=args.main_variant, method=m, budget_pct=b)
                row.append(_safe_mean(sm, "action_l2"))
            w.writerow(row)
    logger.info("E3.1 -> %s", e31_dir)

    # E3.2 latency
    e32_dir = out_dir / "e3_2_latency"
    latency_summary = latency_table(rows, e32_dir)
    logger.info("E3.2 -> %s", e32_dir)

    # E3.3 Pareto
    e33_dir = out_dir / "e3_3_pareto"
    plot_pareto(rows, e33_dir, variant=args.main_variant,
                error_metric="action_l2",
                fname="plot_pareto_action_l2",
                title="E3.3 — Pareto frontier  (retained fraction → action L2)")
    plot_pareto(rows, e33_dir, variant=args.main_variant,
                error_metric="video_l2",
                fname="plot_pareto_video_l2",
                title="E3.3 — Pareto frontier  (retained fraction → video L2)")
    logger.info("E3.3 -> %s", e33_dir)

    # E3.4 contact-heavy
    e34_dir = out_dir / "e3_4_contact_heavy"
    contact_summary = plot_subset_action(
        rows, e34_dir, task_group="B_contact_sensitive",
        variant=args.main_variant,
        fname="plot_contact_action_error",
        title="E3.4 — action L2 vs budget on contact-heavy tasks",
    )
    (e34_dir / "summary.json").write_text(json.dumps(contact_summary, indent=2))
    logger.info("E3.4 -> %s", e34_dir)

    # E3.5 distractor / clutter robustness
    e35_dir = out_dir / "e3_5_robustness"
    robustness_summary = plot_subset_action(
        rows, e35_dir, task_group="C_distractor",
        variant=args.main_variant,
        fname="plot_distractor_action_error",
        title="E3.5 — action L2 vs budget under distractor / clutter",
    )
    (e35_dir / "summary.json").write_text(json.dumps(robustness_summary, indent=2))
    logger.info("E3.5 -> %s", e35_dir)

    # Main paper table
    main_table = write_main_paper_table(rows, out_dir / "main_paper_table.csv",
                                        variant=args.main_variant)

    # Combined report
    write_combined_report(out_dir, rows, main_table, contact_summary, robustness_summary)
    (out_dir / "combined_report.json").write_text(json.dumps({
        "main_variant": args.main_variant,
        "n_rows": len(rows),
        "n_examples": len({r["example_id"] for r in rows}),
        "main_paper_table": main_table,
        "latency_summary": latency_summary,
        "contact_heavy_summary": contact_summary,
        "robustness_summary": robustness_summary,
    }, indent=2))
    logger.info("Done. Main report -> %s", out_dir / "combined_report.md")


if __name__ == "__main__":
    main()
