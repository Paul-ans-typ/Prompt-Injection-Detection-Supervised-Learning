"""
Results aggregation and comparison across all trained models.

Discovers all available metrics.json files under results/, merges them into
a single DataFrame, then produces comparison tables and plots.

Usage:
  python src/compare_results.py                    # all models, all splits
  python src/compare_results.py --split test       # focus on one split
  python src/compare_results.py --metric roc_auc   # change primary metric

Outputs saved to results/comparison/:
  summary_table.csv          - full metrics for every model × split
  summary_table.tex          - LaTeX table (paste directly into paper)
  f1_by_split.png            - grouped bar chart: F1 per split per model
  auc_by_split.png           - grouped bar chart: ROC-AUC per split per model
  heatmap_f1.png             - heatmap: models × splits coloured by F1
  heatmap_auc.png            - heatmap: models × splits coloured by ROC-AUC
  radar_<split>.png          - radar chart: multi-metric comparison per split
  generalization_gap.png     - in-dist test F1 vs OOD test F1 per model
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
COMPARE_DIR  = RESULTS_DIR / "comparison"

# ---------------------------------------------------------------------------
# Display names  (kept separate so plots stay clean)
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    # Baseline
    "metrics_simple":   "TF-IDF + LR (simple)",
    "metrics_enhanced": "TF-IDF + LR (enhanced)",
    # RoBERTa
    "roberta-base":     "RoBERTa-base",
    "roberta-large":    "RoBERTa-large",
}

SPLIT_DISPLAY = {
    "val":           "Validation",
    "test":          "Test (in-dist)",
    "test_deepset":  "Deepset (OOD)",
    "test_wildcard": "Wildcard (OOD)",
}

# Canonical model order for plots (baseline → transformer)
MODEL_ORDER = [
    "TF-IDF + LR (simple)",
    "TF-IDF + LR (enhanced)",
    "RoBERTa-base",
    "RoBERTa-large",
]

SPLIT_ORDER = ["Validation", "Test (in-dist)", "Deepset (OOD)", "Wildcard (OOD)"]

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
METRIC_LABELS = {
    "accuracy":      "Accuracy",
    "precision":     "Precision",
    "recall":        "Recall",
    "f1":            "F1 Score",
    "roc_auc":       "ROC-AUC",
    "avg_precision": "Avg Precision (PR-AUC)",
}

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_metrics() -> list[tuple[str, Path]]:
    """
    Walk results/ and return (model_key, metrics_json_path) for every
    metrics JSON found.  Handles the different directory layouts used by
    baseline and roberta scripts.
    """
    found = []

    # --- Baseline: results/baseline/metrics_simple.json etc.
    baseline_dir = RESULTS_DIR / "baseline"
    if baseline_dir.exists():
        for p in sorted(baseline_dir.glob("metrics_*.json")):
            key = p.stem   # e.g. "metrics_simple"
            found.append((key, p))

    # --- RoBERTa: results/roberta/<model-tag>/metrics.json
    roberta_dir = RESULTS_DIR / "roberta"
    if roberta_dir.exists():
        for p in sorted(roberta_dir.glob("*/metrics.json")):
            key = p.parent.name   # e.g. "roberta-large"
            found.append((key, p))

    return found


def load_metrics(model_key: str, path: Path) -> list[dict]:
    with open(path) as f:
        records = json.load(f)
    display_name = MODEL_DISPLAY.get(model_key, model_key)
    for r in records:
        r["model_key"]    = model_key
        r["model"]        = display_name
        r["split_raw"]    = r.get("split", "unknown")
        r["split"]        = SPLIT_DISPLAY.get(r["split_raw"], r["split_raw"])
    return records


# ---------------------------------------------------------------------------
# Build master DataFrame
# ---------------------------------------------------------------------------

def build_master(found: list[tuple[str, Path]]) -> pd.DataFrame:
    all_rows = []
    for key, path in found:
        try:
            rows = load_metrics(key, path)
            all_rows.extend(rows)
            log.info(f"  Loaded {key:35s} ({len(rows)} splits)")
        except Exception as e:
            log.warning(f"  Could not load {path}: {e}")

    if not all_rows:
        log.error("No metrics files found. Run the training scripts first.")
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)

    # Drop columns we don't need for comparison
    drop_cols = ["report", "split_raw", "model_key", "n_samples"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Enforce canonical ordering (rows present in ORDER lists come first)
    df["model_ord"] = df["model"].apply(
        lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)
    )
    df["split_ord"] = df["split"].apply(
        lambda s: SPLIT_ORDER.index(s) if s in SPLIT_ORDER else len(SPLIT_ORDER)
    )
    df = df.sort_values(["model_ord", "split_ord"]).drop(columns=["model_ord", "split_ord"])
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "summary_table.csv"
    cols = ["model", "split"] + [m for m in METRICS if m in df.columns]
    df[cols].to_csv(path, index=False, float_format="%.4f")
    log.info(f"CSV saved: {path}")


def save_latex(df: pd.DataFrame, primary_metric: str, out_dir: Path) -> None:
    """
    Pivot table: rows = models, columns = splits, values = primary_metric.
    Bold the best value in each column.
    """
    if primary_metric not in df.columns:
        return

    pivot = df.pivot_table(
        index="model", columns="split", values=primary_metric, aggfunc="mean"
    )

    # Reorder rows/cols to canonical order
    row_order = [m for m in MODEL_ORDER if m in pivot.index]
    col_order = [s for s in SPLIT_ORDER if s in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    # Build LaTeX manually so we can bold best values per column
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{" + METRIC_LABELS.get(primary_metric, primary_metric) +
        r" comparison across models and evaluation sets}"
    )
    lines.append(r"\label{tab:results}")
    col_spec = "l" + "c" * len(pivot.columns)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header = "Model & " + " & ".join(pivot.columns.tolist()) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best per column
    best = pivot.max(axis=0)

    for model_name, row in pivot.iterrows():
        cells = []
        for col in pivot.columns:
            val = row[col]
            if pd.isna(val):
                cells.append("—")
            elif abs(val - best[col]) < 1e-4:
                cells.append(r"\textbf{" + f"{val:.4f}" + "}")
            else:
                cells.append(f"{val:.4f}")
        lines.append(model_name + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = out_dir / "summary_table.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"LaTeX table saved: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

PALETTE = "tab10"


def _present_models(df: pd.DataFrame) -> list[str]:
    """Return models in canonical order, filtered to those in df."""
    present = set(df["model"].unique())
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered += sorted(present - set(MODEL_ORDER))
    return ordered


def plot_grouped_bars(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
) -> None:
    """Grouped bar chart: x = split, groups = model, y = metric value."""
    if metric not in df.columns:
        return

    models = _present_models(df)
    splits = [s for s in SPLIT_ORDER if s in df["split"].unique()]

    x     = np.arange(len(splits))
    width = 0.8 / max(len(models), 1)
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(max(9, len(splits) * 2.5), 5))
    colors = plt.cm.get_cmap(PALETTE, len(models)).colors

    for i, (model, offset) in enumerate(zip(models, offsets)):
        values = []
        for split in splits:
            row = df[(df["model"] == model) & (df["split"] == split)]
            values.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
        bars = ax.bar(x + offset, values, width * 0.9, label=model, color=colors[i])

        # Annotate bars with value
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Evaluation Split and Model")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_by_split.png", dpi=150)
    plt.close(fig)
    log.info(f"Bar chart saved: {metric}_by_split.png")


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
) -> None:
    if metric not in df.columns:
        return

    models = _present_models(df)
    splits = [s for s in SPLIT_ORDER if s in df["split"].unique()]

    pivot = pd.DataFrame(index=models, columns=splits, dtype=float)
    for model in models:
        for split in splits:
            row = df[(df["model"] == model) & (df["split"] == split)]
            pivot.loc[model, split] = float(row[metric].iloc[0]) if not row.empty else np.nan

    fig, ax = plt.subplots(figsize=(max(6, len(splits) * 1.8), max(4, len(models) * 0.7)))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        vmin=max(0, float(pivot.min().min()) - 0.05),
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
    )
    ax.set_title(
        f"{METRIC_LABELS.get(metric, metric)} — All Models × All Evaluation Sets",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=150)
    plt.close(fig)
    log.info(f"Heatmap saved: heatmap_{metric}.png")


def plot_radar(
    df: pd.DataFrame,
    split_name: str,
    out_dir: Path,
) -> None:
    """Spider/radar chart comparing all metrics for all models on one split."""
    radar_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    radar_metrics = [m for m in radar_metrics if m in df.columns]
    if len(radar_metrics) < 3:
        return

    sub = df[df["split"] == split_name]
    if sub.empty:
        return

    models = _present_models(sub)
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = plt.cm.get_cmap(PALETTE, len(models)).colors

    for i, model in enumerate(models):
        row = sub[sub["model"] == model]
        if row.empty:
            continue
        values = [float(row[m].iloc[0]) for m in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.8, color=colors[i], label=model)
        ax.fill(angles, values, alpha=0.07, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in radar_metrics], fontsize=9
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(f"Multi-Metric Radar — {split_name}", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()

    safe_name = split_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    fig.savefig(out_dir / f"radar_{safe_name}.png", dpi=150)
    plt.close(fig)
    log.info(f"Radar chart saved: radar_{safe_name}.png")


def plot_generalization_gap(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter plot: x = in-distribution test F1, y = OOD test F1 (wildcard).
    Models that generalise well cluster near the diagonal.
    """
    if "f1" not in df.columns:
        return

    in_dist = df[df["split"] == "Test (in-dist)"][["model", "f1"]].rename(
        columns={"f1": "f1_indist"}
    )
    ood = df[df["split"] == "Wildcard (OOD)"][["model", "f1"]].rename(
        columns={"f1": "f1_ood"}
    )

    if in_dist.empty or ood.empty:
        log.warning("Skipping generalization gap plot — need both test and test_wildcard splits")
        return

    merged = in_dist.merge(ood, on="model", how="inner")
    if merged.empty:
        return

    models  = _present_models(merged)
    colors  = plt.cm.get_cmap(PALETTE, len(models)).colors
    color_map = {m: c for m, c in zip(models, colors)}

    fig, ax = plt.subplots(figsize=(7, 6))
    lims = [
        min(merged["f1_indist"].min(), merged["f1_ood"].min()) - 0.03,
        max(merged["f1_indist"].max(), merged["f1_ood"].max()) + 0.03,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="No gap (diagonal)")

    for _, row in merged.iterrows():
        color = color_map.get(row["model"], "gray")
        ax.scatter(row["f1_indist"], row["f1_ood"], s=120, color=color, zorder=5)
        ax.annotate(
            row["model"],
            xy=(row["f1_indist"], row["f1_ood"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("F1 — Test (in-distribution)")
    ax.set_ylabel("F1 — Wildcard OOD (real-world prompts)")
    ax.set_title("Generalization Gap: In-Distribution vs OOD Performance")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap.png", dpi=150)
    plt.close(fig)
    log.info("Generalization gap plot saved.")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, primary_metric: str) -> None:
    cols = ["model", "split"] + [m for m in METRICS if m in df.columns]
    sub  = df[cols].copy()

    print("\n" + "=" * 80)
    print(f"  RESULTS SUMMARY  —  sorted by {primary_metric}")
    print("=" * 80)

    for split in SPLIT_ORDER:
        chunk = sub[sub["split"] == split]
        if chunk.empty:
            continue
        chunk = chunk.drop(columns="split").sort_values(primary_metric, ascending=False)
        print(f"\n  {split}")
        print("  " + "-" * 70)
        print(chunk.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and compare model results")
    parser.add_argument(
        "--metric", default="f1",
        choices=METRICS,
        help="Primary metric for table sorting and LaTeX output (default: f1)",
    )
    parser.add_argument(
        "--split", default=None,
        help="If set, print detailed table for this split only "
             "(e.g. 'test', 'test_wildcard')",
    )
    args = parser.parse_args()

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Discover + load --------------------------------------------------
    log.info("Discovering metrics files...")
    found = discover_metrics()

    if not found:
        log.error(
            "No metrics files found under results/. "
            "Run the training scripts first."
        )
        raise SystemExit(1)

    log.info(f"Found {len(found)} result file(s):")
    df = build_master(found)

    # ---- Filter to one split if requested ---------------------------------
    if args.split:
        raw_split = args.split
        display_split = SPLIT_DISPLAY.get(raw_split, raw_split)
        df_view = df[df["split"] == display_split]
        if df_view.empty:
            log.warning(f"No results found for split '{args.split}'")
        else:
            print_summary(df_view, args.metric)
    else:
        print_summary(df, args.metric)

    # ---- Save tables ------------------------------------------------------
    save_csv(df, COMPARE_DIR)
    save_latex(df, args.metric, COMPARE_DIR)

    # ---- Plots ------------------------------------------------------------
    log.info("Generating plots...")

    plot_grouped_bars(df, "f1",      COMPARE_DIR)
    plot_grouped_bars(df, "roc_auc", COMPARE_DIR)
    plot_heatmap(df, "f1",           COMPARE_DIR)
    plot_heatmap(df, "roc_auc",      COMPARE_DIR)

    for split in SPLIT_ORDER:
        plot_radar(df, split, COMPARE_DIR)

    plot_generalization_gap(df, COMPARE_DIR)

    log.info(f"\nAll outputs saved to {COMPARE_DIR}")


if __name__ == "__main__":
    main()
