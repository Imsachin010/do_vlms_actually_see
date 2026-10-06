import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS10_DIR = str(BASE_DIR / "Figures for Paper" / "results10")
PHASE2_CSV    = str(BASE_DIR / "data" / "processed" / "phase2_vlm_results.csv")
PLOTS_DIR     = os.path.join(RESULTS10_DIR, "plots")

MODEL_ORDER = [
    "llama4-scout",
    "SmolVLM2-2.2B",
    "Qwen2.5-VL-7B",
    "Qwen2-VL-2B",
    "LLaVA-v1.6-Mistral-7B",
    "LLaVA-1.5-7B",
    "InstructBLIP-FlanT5XL",
    "Gemma-3-4B",
]

def load_data() -> pd.DataFrame:
    frames = []
    for csv_path in glob.glob(os.path.join(RESULTS10_DIR, "*_results.csv")):
        df = pd.read_csv(csv_path)
        frames.append(df)
    if os.path.exists(PHASE2_CSV):
        df2 = pd.read_csv(PHASE2_CSV)
        # Ensure is_biased exists
        if "is_biased" not in df2.columns and "is_correct" in df2.columns:
            df2["is_biased"] = (1 - df2["is_correct"].astype(int))
        frames.append(df2)
    master = pd.concat(frames, ignore_index=True)
    master["concept"] = master["concept"].str.lower().str.strip()
    master["model"]   = master["model"].str.strip()
    return master


def compute_metrics(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in master.groupby("model"):
        valid     = group.dropna(subset=["predicted", "true_legs"])
        accuracy  = (valid["predicted"].astype(int) == valid["true_legs"].astype(int)).mean() * 100 if len(valid) else 0.0
        bias_rate = group["is_biased"].mean() * 100
        rows.append({"Model": model, "Accuracy (%)": accuracy, "Bias Rate (%)": bias_rate})
    df = pd.DataFrame(rows)
    # Re-order if possible
    order = [m for m in MODEL_ORDER if m in df["Model"].values]
    rest  = [m for m in df["Model"].values if m not in order]
    df = pd.concat([df[df["Model"].isin(order)].set_index("Model").loc[order].reset_index(),
                    df[df["Model"].isin(rest)]], ignore_index=True)
    return df


def plot_bar_chart(metrics_df: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(metrics_df))
    w = 0.38
    bars_acc  = ax.bar(x - w/2, metrics_df["Accuracy (%)"],  w, label="Accuracy",  color="#4e79a7", edgecolor="black", linewidth=0.6)
    bars_bias = ax.bar(x + w/2, metrics_df["Bias Rate (%)"], w, label="Bias Rate", color="#e15759", edgecolor="black", linewidth=0.6)

    # Value labels on bars
    for bar in list(bars_acc) + list(bars_bias):
        v = bar.get_height()
        if v > 2:
            ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("VLM Benchmark: Accuracy vs Prior Bias Rate", fontsize=15, fontweight="bold", pad=18)
    ax.legend(frameon=False, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "accuracy_vs_bias_bar.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved bar chart → {out}")


def plot_scatter(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.RdYlGn_r
    colors = cmap(metrics_df["Bias Rate (%)"].values / 100)
    sc = ax.scatter(metrics_df["Accuracy (%)"], metrics_df["Bias Rate (%)"],
                    c=metrics_df["Bias Rate (%)"], cmap="RdYlGn_r",
                    s=220, edgecolors="black", linewidths=0.7, vmin=0, vmax=100, zorder=3)
    for _, row in metrics_df.iterrows():
        ax.annotate(row["Model"], (row["Accuracy (%)"], row["Bias Rate (%)"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8.5, color="#333")
    plt.colorbar(sc, ax=ax, label="Bias Rate (%)")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("Bias Rate (%)", fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title("Accuracy vs Bias Rate — VLM Benchmark", fontsize=15, fontweight="bold", pad=18)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "accuracy_vs_bias_scatter.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved scatter plot → {out}")


def plot_heatmap(master: pd.DataFrame):
    """Bias Rate % heatmap: models × concepts."""
    pivot = master.groupby(["model", "concept"])["is_biased"].mean().mul(100).unstack(fill_value=np.nan)
    pivot = pivot.sort_index(axis=0)  # sort models
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)  # sort concepts

    h = max(6, len(pivot) * 0.65)
    w = max(10, len(pivot.columns) * 0.75)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=0, vmax=100, linewidths=0.4, linecolor="#eee",
                cbar_kws={"label": "Bias Rate (%)", "shrink": 0.8})
    ax.set_title("Bias Rate (%) per Model × Concept", fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("Concept", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "bias_heatmap_model_concept.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved heatmap → {out}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    master = load_data()
    metrics_df = compute_metrics(master)
    print(metrics_df.to_string(index=False))

    plot_bar_chart(metrics_df)
    plot_scatter(metrics_df)
    plot_heatmap(master)
    print("\n✅ All plots saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()
