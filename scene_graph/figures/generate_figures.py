"""
Generate publication-quality figures for the Scene Graph vs Raw Vision comparison.

Outputs:
  - fig1_sg_vs_raw_accuracy.pdf  (Bar chart: Raw vs SG accuracy per model)
  - fig2_error_breakdown.pdf     (Stacked bar: perception vs reasoning error)
  - fig3_bias_comparison.pdf     (Semantic bias rate: raw vs SG)
  - fig4_perception_pipeline.pdf (Flow diagram showing where bias originates)
  - fig5_confusion_matrix.pdf    (Confusion matrix for raw image predictions)
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Styling (NeurIPS / CVPR quality) ──────────────────────────────────

sns.set_theme(
    style="whitegrid",
    font_scale=1.2,
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    },
)

COLORS = {
    "raw": "#E74C3C",       # Red for raw vision
    "sg": "#2ECC71",        # Green for scene graph
    "correct": "#2ECC71",
    "perception": "#E67E22", # Orange
    "reasoning": "#9B59B6",  # Purple
    "bias_sg": "#2ECC71",
    "bias_raw": "#E74C3C",
    "prior": "#3498DB",     # Blue for "answered 4"
    "other": "#95A5A6",     # Gray
}

OUTPUT_DIR = Path("scene_graph/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data paths (relative to project root) ─────────────────────────────
SG_RESULTS_DIR = Path("scene_graph/results")
RAW_RESULTS_DIR = Path("data/raw/results_hf")

# ── Data Loading ──────────────────────────────────────────────────────

def load_data():
    """Load raw vision and scene graph results."""
    # Scene graph results
    sg_files = {
        "Gemma-4-31B": "gemma_4_31b_sg_results.csv",
        "Qwen3-VL-235B-A22B": "qwen3_vl_235b_a22b_sg_results.csv",
        "Llama-4-Scout": "llama_4_scout_sg_results.csv",
    }

    # Raw vision results from main experiment
    raw_files = {
        "Gemma-4-31B": "gemma_4_31b_results.csv",
        "Qwen3-VL-235B-A22B": "qwen3_vl_235b_a22b_results.csv",
        "Llama-4-Scout": "llama_4_scout_results.csv",
    }

    sg_data = {}
    raw_data = {}

    for model_name, fname in sg_files.items():
        path = SG_RESULTS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            df["true_legs"] = pd.to_numeric(df["true_legs"], errors="coerce")
            df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
            sg_data[model_name] = df[df["status"] == "ok"]

    for model_name, fname in raw_files.items():
        path = RAW_RESULTS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            df["true_legs"] = pd.to_numeric(df["true_legs"], errors="coerce")
            df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
            # Extract image_id from filename
            df["image_id"] = df["image"].apply(
                lambda x: str(x).rsplit(".", 1)[0] if pd.notna(x) else x
            )
            raw_data[model_name] = df[df["status"] == "ok"]

    return sg_data, raw_data


# ── Figure 1: Raw vs Scene Graph Accuracy ─────────────────────────────

def fig1_sg_vs_raw_accuracy(sg_data, raw_data):
    """Side-by-side bar chart comparing accuracy per model."""
    models = sorted(sg_data.keys())
    
    raw_acc = []
    sg_acc = []
    raw_err = []
    sg_err = []

    for m in models:
        if m in raw_data and len(raw_data[m]) > 0:
            acc = (raw_data[m]["predicted"] == raw_data[m]["true_legs"]).mean()
            raw_acc.append(acc)
            raw_err.append(np.sqrt(acc * (1 - acc) / len(raw_data[m])))
        else:
            raw_acc.append(0)
            raw_err.append(0)

        if m in sg_data and len(sg_data[m]) > 0:
            acc = (sg_data[m]["predicted"] == sg_data[m]["true_legs"]).mean()
            sg_acc.append(acc)
            sg_err.append(np.sqrt(acc * (1 - acc) / len(sg_data[m])))
        else:
            sg_acc.append(0)
            sg_err.append(0)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, raw_acc, width, label="Raw Vision",
                   color=COLORS["raw"], edgecolor="white", linewidth=0.5,
                   yerr=raw_err, capsize=4, error_kw={"linewidth": 1.2})
    bars2 = ax.bar(x + width / 2, sg_acc, width, label="Scene Graph",
                   color=COLORS["sg"], edgecolor="white", linewidth=0.5,
                   yerr=sg_err, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_title("Raw Vision vs Scene Graph Accuracy", fontweight="bold", fontsize=13)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    sns.despine()
    fig.tight_layout()
    path = OUTPUT_DIR / "fig1_sg_vs_raw_accuracy.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 2: Error Breakdown ─────────────────────────────────────────

def fig2_error_breakdown(sg_data, raw_data):
    """Stacked bar: perception error, reasoning error, both correct."""
    models = sorted(sg_data.keys())

    perception = []
    reasoning = []
    both_correct = []
    both_wrong = []

    for m in models:
        if m not in raw_data or len(raw_data[m]) == 0:
            perception.append(0)
            reasoning.append(0)
            both_correct.append(0)
            both_wrong.append(0)
            continue

                # Merge raw and SG on image_id
        # Both have 'true_legs' column; join preserves one copy
        merged = raw_data[m][["image_id", "true_legs", "predicted"]].merge(
            sg_data[m][["image_id", "predicted"]],
            on="image_id",
            suffixes=("_raw", "_sg"),
            how="inner",
        )
        merged = merged.dropna()
        if len(merged) == 0:
            perception.append(0)
            reasoning.append(0)
            both_correct.append(0)
            both_wrong.append(0)
            continue

        correct_raw = (merged["predicted_raw"] == merged["true_legs"])
        correct_sg = (merged["predicted_sg"] == merged["true_legs"])

        n = len(merged)
        perception.append((~correct_raw & correct_sg).sum() / n)
        reasoning.append((correct_raw & ~correct_sg).sum() / n)
        both_correct.append((correct_raw & correct_sg).sum() / n)
        both_wrong.append((~correct_raw & ~correct_sg).sum() / n)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(models))
    width = 0.5

    ax.bar(x, both_correct, width, label="Both Correct",
           color=COLORS["correct"], edgecolor="white", linewidth=0.5)
    ax.bar(x, perception, width, bottom=both_correct, label="Perception Error",
           color=COLORS["perception"], edgecolor="white", linewidth=0.5)
    ax.bar(x, reasoning, width, bottom=np.array(both_correct) + np.array(perception),
           label="Reasoning Error", color=COLORS["reasoning"], edgecolor="white", linewidth=0.5)
    ax.bar(x, both_wrong, width,
           bottom=np.array(both_correct) + np.array(perception) + np.array(reasoning),
           label="Both Wrong", color=COLORS["other"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Proportion of Samples", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.set_title("Error Source Decomposition", fontweight="bold", fontsize=13)

    sns.despine()
    fig.tight_layout()
    path = OUTPUT_DIR / "fig2_error_breakdown.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 3: Semantic Bias Rate ──────────────────────────────────────

def fig3_bias_comparison(sg_data, raw_data):
    """Grouped bar: how often each model answered '4' (semantic prior)."""
    models = sorted(sg_data.keys())

    raw_bias = []
    sg_bias = []

    for m in models:
        # Raw: proportion of OK responses that answered "4"
        if m in raw_data and len(raw_data[m]) > 0:
            rb = (raw_data[m]["predicted"] == 4).mean()
        else:
            rb = 0
        raw_bias.append(rb)

        # SG: proportion of OK responses that answered "4"
        if m in sg_data and len(sg_data[m]) > 0:
            sb = (sg_data[m]["predicted"] == 4).mean()
        else:
            sb = 0
        sg_bias.append(sb)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, raw_bias, width, label="Raw Vision",
                   color=COLORS["bias_raw"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, sg_bias, width, label="Scene Graph",
                   color=COLORS["bias_sg"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel('Predicted "4 Legs" Rate', fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_title("Semantic Prior Collapse Rate", fontweight="bold", fontsize=13)

    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    sns.despine()
    fig.tight_layout()
    path = OUTPUT_DIR / "fig3_bias_comparison.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 4: Confusion Matrix (Raw Vision) ───────────────────────────

def fig4_confusion_matrix(raw_data):
    """Confusion matrix showing ground truth vs predicted leg counts on raw images."""
    # Aggregate all models
    all_raw = []
    for m_name, df in raw_data.items():
        if len(df) > 0:
            d = df[["true_legs", "predicted"]].copy()
            d["model"] = m_name
            all_raw.append(d)

    if not all_raw:
        print("  WARN: No raw data for confusion matrix")
        return

    combined = pd.concat(all_raw, ignore_index=True)
    combined = combined.dropna()
    combined["true_legs"] = combined["true_legs"].astype(int)
    combined["predicted"] = combined["predicted"].astype(int)

    # Ensure 3,4,5 are in the matrix
    all_vals = sorted(set(combined["true_legs"].unique()) | set(combined["predicted"].unique()))

    fig, axes = plt.subplots(1, len(raw_data), figsize=(5 * len(raw_data), 4.5), sharey=True)
    if len(raw_data) == 1:
        axes = [axes]

    for ax, (m_name, df) in zip(axes, raw_data.items()):
        d = df.dropna(subset=["true_legs", "predicted"])
        d["true_legs"] = d["true_legs"].astype(int)
        d["predicted"] = d["predicted"].astype(int)

        cm = pd.crosstab(
            d["true_legs"], d["predicted"],
            rownames=["True"], colnames=["Predicted"],
            normalize="index",
        )

        # Reindex to include 3,4,5
        cm = cm.reindex(index=all_vals, columns=all_vals, fill_value=0)

        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    ax=ax, cbar_kws={"label": "Proportion"},
                    vmin=0, vmax=1, linewidths=0.5, linecolor="white")
        ax.set_title(m_name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted Legs", fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("True Legs", fontweight="bold")

    fig.suptitle("Raw Vision: Confusion Matrix (Ground Truth vs Prediction)",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig4_confusion_matrix.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 5: Perception Pipeline Diagram ─────────────────────────────

def fig5_perception_pipeline():
    """
    Conceptual diagram showing where bias originates.
    Top: Image → VLM → Perception fails → Prior collapse → Answer "4"
    Bottom: Scene Graph → VLM → No ambiguity → Correct answer
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5))

    # Parameters
    box_style = dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="0.3", linewidth=1.5)
    arrow_style = dict(arrowstyle="->", color="0.3", linewidth=1.5)

    # ── Top row: Raw Vision (failure) ──
    ax1.text(0.02, 0.6, "Input:", fontsize=10, fontweight="bold", va="center",
             transform=ax1.transAxes)
    ax1.text(0.12, 0.6, "Counterfactual\nImage", fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["raw"],
                       edgecolor="none", alpha=0.15), transform=ax1.transAxes)

    ax1.annotate("", xy=(0.22, 0.6), xytext=(0.18, 0.6), arrowprops=arrow_style,
                 transform=ax1.transAxes)

    ax1.text(0.24, 0.6, "VLM\nVision", fontsize=9, ha="center", va="center",
             bbox=box_style, transform=ax1.transAxes)

    ax1.annotate("", xy=(0.34, 0.6), xytext=(0.30, 0.6), arrowprops=arrow_style,
                 transform=ax1.transAxes)

    ax1.text(0.40, 0.6, "Perception\nFail (fail)", fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["perception"],
                       edgecolor="none", alpha=0.25), transform=ax1.transAxes)

    ax1.annotate("", xy=(0.56, 0.6), xytext=(0.46, 0.6), arrowprops=arrow_style,
                 transform=ax1.transAxes)

    ax1.text(0.62, 0.6, 'Prior\nOverride\n"4 legs"', fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["prior"],
                       edgecolor="none", alpha=0.2), transform=ax1.transAxes)

    ax1.annotate("", xy=(0.78, 0.6), xytext=(0.70, 0.6), arrowprops=arrow_style,
                 transform=ax1.transAxes)

    ax1.text(0.85, 0.6, "Wrong\nAnswer", fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["raw"],
                       edgecolor=COLORS["raw"], alpha=0.2), transform=ax1.transAxes)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("(a) Raw Vision Pipeline", fontweight="bold", fontsize=11, loc="left")

    # ── Bottom row: Scene Graph (success) ──
    ax2.text(0.02, 0.6, "Input:", fontsize=10, fontweight="bold", va="center",
             transform=ax2.transAxes)
    ax2.text(0.12, 0.6, "Structured\nScene Graph", fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["sg"],
                       edgecolor="none", alpha=0.15), transform=ax2.transAxes)

    ax2.annotate("", xy=(0.22, 0.6), xytext=(0.18, 0.6), arrowprops=arrow_style,
                 transform=ax2.transAxes)

    ax2.text(0.24, 0.6, "VLM\nText", fontsize=9, ha="center", va="center",
             bbox=box_style, transform=ax2.transAxes)

    ax2.annotate("", xy=(0.38, 0.6), xytext=(0.30, 0.6), arrowprops=arrow_style,
                 transform=ax2.transAxes)

    ax2.text(0.44, 0.6, "Perception\nOK (ok)", fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["sg"],
                       edgecolor="none", alpha=0.25), transform=ax2.transAxes)

    ax2.annotate("", xy=(0.60, 0.6), xytext=(0.50, 0.6), arrowprops=arrow_style,
                 transform=ax2.transAxes)

    ax2.text(0.66, 0.6, 'Strict\nReasoning\n"3 legs"', fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["sg"],
                       edgecolor="none", alpha=0.2), transform=ax2.transAxes)

    ax2.annotate("", xy=(0.82, 0.6), xytext=(0.74, 0.6), arrowprops=arrow_style,
                 transform=ax2.transAxes)

    ax2.text(0.88, 0.6, "Correct\nAnswer (ok)", fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["sg"],
                       edgecolor=COLORS["sg"], alpha=0.2), transform=ax2.transAxes)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("(b) Scene Graph Pipeline", fontweight="bold", fontsize=11, loc="left")

    fig.tight_layout()
    path = OUTPUT_DIR / "fig5_perception_pipeline.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 6: Accuracy by Background (Raw Vision only) ────────────────

def fig6_accuracy_by_background(raw_data):
    """Show how accuracy on raw images varies by background type."""
    models = sorted(raw_data.keys())

    plain_acc = []
    natural_acc = []
    plain_err = []
    natural_err = []

    for m in models:
        df = raw_data[m].dropna(subset=["true_legs", "predicted", "background"])
        if len(df) == 0:
            plain_acc.append(0)
            natural_acc.append(0)
            plain_err.append(0)
            natural_err.append(0)
            continue

        for bg, acc_list, err_list in [
            ("plain", plain_acc, plain_err),
            ("natural", natural_acc, natural_err),
        ]:
            sub = df[df["background"] == bg]
            if len(sub) > 0:
                acc = (sub["predicted"] == sub["true_legs"]).mean()
                acc_list.append(acc)
                err_list.append(np.sqrt(acc * (1 - acc) / len(sub)))
            else:
                acc_list.append(0)
                err_list.append(0)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, plain_acc, width, label="Plain Background",
                   color="#3498DB", edgecolor="white", linewidth=0.5,
                   yerr=plain_err, capsize=4, error_kw={"linewidth": 1.2})
    bars2 = ax.bar(x + width / 2, natural_acc, width, label="Natural Background",
                   color="#E67E22", edgecolor="white", linewidth=0.5,
                   yerr=natural_err, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_ylabel("Raw Image Accuracy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_title("Raw Vision Accuracy by Background Type", fontweight="bold", fontsize=13)

    sns.despine()
    fig.tight_layout()
    path = OUTPUT_DIR / "fig6_accuracy_by_background.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 7: Summary Heatmap ─────────────────────────────────────────

def fig7_summary_heatmap(sg_data, raw_data):
    """Heatmap summarizing all metrics across models."""
    models = sorted(sg_data.keys())
    metrics_list = []

    for m in models:
        # Raw metrics
        if m in raw_data and len(raw_data[m]) > 0:
            df = raw_data[m].dropna(subset=["true_legs", "predicted"])
            acc_raw = (df["predicted"] == df["true_legs"]).mean()
            bias_raw = (df["predicted"] == 4).mean()
        else:
            acc_raw = 0
            bias_raw = 0

        # SG metrics
        if m in sg_data and len(sg_data[m]) > 0:
            df = sg_data[m].dropna(subset=["true_legs", "predicted"])
            acc_sg = (df["predicted"] == df["true_legs"]).mean()
            bias_sg = (df["predicted"] == 4).mean()
            reason_err = (df["error_type"] == "reasoning_error").mean()
        else:
            acc_sg = 0
            bias_sg = 0
            reason_err = 0

        metrics_list.append({
            "Model": m,
            "Raw\nAccuracy": acc_raw,
            "SG\nAccuracy": acc_sg,
            "Raw\nBias": bias_raw,
            "SG\nBias": bias_sg,
            "Perception\nError": acc_sg - acc_raw,
            "Reasoning\nError": reason_err,
        })

    metrics_df = pd.DataFrame(metrics_list)
    if "Model" in metrics_df.columns:
        metrics_df = metrics_df.set_index("Model")
    else:
        # Use first column if no explicit Model column
        metrics_df = metrics_df.set_index(metrics_df.columns[0])
    metrics_df = metrics_df * 100  # convert to percentage

    fig, ax = plt.subplots(figsize=(8, 3.5))

    sns.heatmap(metrics_df, annot=True, fmt=".0f", cmap="RdYlGn",
                ax=ax, cbar_kws={"label": "Percentage (%)", "shrink": 0.8},
                vmin=0, vmax=100, linewidths=1, linecolor="white",
                annot_kws={"fontsize": 10, "fontweight": "bold"})
    ax.set_title("Model Performance Summary: Raw Vision vs Scene Graph",
                 fontweight="bold", fontsize=12, pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("Metric", fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "fig7_summary_heatmap.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def run():
    print("Loading data...")
    sg_data, raw_data = load_data()
    print(f"  Scene graph models: {list(sg_data.keys())}")
    print(f"  Raw vision models:  {list(raw_data.keys())}")

    print("\nGenerating figures...")

    print("\n[Fig 1] Raw vs Scene Graph Accuracy")
    fig1_sg_vs_raw_accuracy(sg_data, raw_data)

    print("\n[Fig 2] Error Source Decomposition")
    fig2_error_breakdown(sg_data, raw_data)

    print("\n[Fig 3] Semantic Prior Collapse Rate")
    fig3_bias_comparison(sg_data, raw_data)

    print("\n[Fig 4] Confusion Matrix (Raw Vision)")
    fig4_confusion_matrix(raw_data)

    print("\n[Fig 5] Perception Pipeline Diagram")
    fig5_perception_pipeline()

    print("\n[Fig 6] Accuracy by Background")
    fig6_accuracy_by_background(raw_data)

    print("\n[Fig 7] Summary Heatmap")
    fig7_summary_heatmap(sg_data, raw_data)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("  Format: PDF (vector graphics, publication-ready)")


if __name__ == "__main__":
    run()
