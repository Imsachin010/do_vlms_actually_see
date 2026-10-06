"""
salience_bias_correlation.py
─────────────────────────────
Spearman correlation between per-concept visual salience scores
and per-concept bias rates for Llama-4-Scout (and any other model
present in the results CSVs).

Usage:
    python src/analysis/salience_bias_correlation.py
    python src/analysis/salience_bias_correlation.py --model "llama4-scout"
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import spearmanr


# ─────────────────────────────────────────────
# Visual salience scores (higher = more visually
# distinctive / commonly seen concept)
# ─────────────────────────────────────────────
SALIENCE = {
    "elephant":  1.00,
    "lion":      0.85,
    "deer":      0.80,
    "horse":     0.75,
    "cheetah":   0.60,
    "leopard":   0.55,
    "fox":       0.50,
    "hyena":     0.45,
    "cat":       0.40,
    "dog":       0.35,
    "serval":    0.30,
    "binturong": 0.25,
    "okapi":     0.22,
    "fossa":     0.20,
    "numbat":    0.18,
}

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIRS = [
    str(BASE_DIR / "Figures for Paper" / "results10"),
    str(BASE_DIR / "data" / "processed"),
]
PHASE2_CSV = str(BASE_DIR / "data" / "processed" / "phase2_vlm_results.csv")
OUTPUT_DIR = str(BASE_DIR / "Figures for Paper" / "results10" / "plots")


def load_all_results() -> pd.DataFrame:
    """Load all CSVs from results10 + phase2 into one DataFrame."""
    frames = []

    # results10 CSVs
    csv_files = glob.glob(os.path.join(RESULTS_DIRS[0], "*_results.csv"))
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)

    # phase2 (Llama-4-Scout via Groq API)
    if os.path.exists(PHASE2_CSV):
        df2 = pd.read_csv(PHASE2_CSV)
        # normalise column names to match results10 schema
        if "is_correct" in df2.columns and "is_biased" not in df2.columns:
            df2["is_biased"] = (~df2["is_correct"].astype(bool)).astype(int)
        frames.append(df2)

    if not frames:
        raise FileNotFoundError("No result CSVs found. Check RESULTS_DIRS and PHASE2_CSV paths.")

    master = pd.concat(frames, ignore_index=True)
    master["concept"] = master["concept"].str.lower().str.strip()
    return master


def run_salience_analysis(df: pd.DataFrame, model_filter: str = None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Available models
    available_models = sorted(df["model"].unique())
    print(f"\nAvailable models: {available_models}\n")

    if model_filter:
        target_models = [m for m in available_models if model_filter.lower() in m.lower()]
        if not target_models:
            print(f"❌ No model matching '{model_filter}' found. Available: {available_models}")
            return
    else:
        target_models = available_models

    # ──────────────────────────────────────────
    # 1. Per-model Spearman correlation
    # ──────────────────────────────────────────
    print("=" * 65)
    print(f"{'Model':<35} {'ρ':>7}  {'p-value':>10}  {'Result'}")
    print("=" * 65)

    summary_rows = []
    for model in target_models:
        mdf = df[df["model"] == model]
        bias_by_concept = mdf.groupby("concept")["is_biased"].mean()

        # Only keep concepts that have salience scores
        valid = {c: bias_by_concept[c] for c in bias_by_concept.index if c in SALIENCE}
        if len(valid) < 5:
            print(f"{model:<35} {'—':>7}  {'—':>10}  Not enough concepts ({len(valid)})")
            continue

        concepts    = list(valid.keys())
        bias_vals   = [valid[c] for c in concepts]
        sal_vals    = [SALIENCE[c] for c in concepts]

        rho, p = spearmanr(sal_vals, bias_vals)

        if p < 0.05 and abs(rho) > 0.5:
            label = "✅ STRONG"
        elif p < 0.05 and abs(rho) > 0.3:
            label = "⚠️  MEDIUM"
        else:
            label = "❌ NULL"

        print(f"{model:<35} {rho:>7.3f}  {p:>10.4f}  {label}")

        summary_rows.append({
            "model":    model,
            "rho":      rho,
            "p_value":  p,
            "n_concepts": len(concepts),
        })

        # ──────────────────────────────────────────
        # 2. Per-concept breakdown
        # ──────────────────────────────────────────
        concept_df = pd.DataFrame({
            "concept":   concepts,
            "bias_rate": [v * 100 for v in bias_vals],
            "salience":  sal_vals,
        }).sort_values("salience", ascending=False)

        print(f"\n  Per-concept breakdown for {model}:")
        print(f"  {'Concept':<14} {'Salience':>10}  {'Bias Rate (%)':>13}")
        print(f"  {'─'*40}")
        for _, row in concept_df.iterrows():
            print(f"  {row['concept']:<14} {row['salience']:>10.2f}  {row['bias_rate']:>12.1f}%")
        print()

        # ──────────────────────────────────────────
        # 3. Scatter plot: Salience vs Bias Rate
        # ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.set_theme(style="whitegrid")

        color_map = concept_df["bias_rate"].values
        sc = ax.scatter(concept_df["salience"], concept_df["bias_rate"],
                        c=color_map, cmap="RdYlGn_r", s=160,
                        edgecolors="black", linewidths=0.7, zorder=3)

        # Regression line
        z = np.polyfit(concept_df["salience"], concept_df["bias_rate"], 1)
        xline = np.linspace(0, 1.05, 100)
        ax.plot(xline, np.polyval(z, xline), "--", color="#555", linewidth=1.4, alpha=0.6, zorder=2)

        # Labels
        for _, row in concept_df.iterrows():
            ax.annotate(row["concept"].capitalize(),
                        (row["salience"], row["bias_rate"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, color="#333")

        plt.colorbar(sc, ax=ax, label="Bias Rate (%)")

        ax.set_title(
            f"Visual Salience vs Bias Rate — {model}\n"
            f"Spearman ρ = {rho:.3f},  p = {p:.4f}",
            fontsize=13, fontweight="bold", pad=15
        )
        ax.set_xlabel("Visual Salience Score", fontsize=11)
        ax.set_ylabel("Bias Rate (%)", fontsize=11)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        plt.tight_layout()

        safe_name = model.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = os.path.join(OUTPUT_DIR, f"salience_scatter_{safe_name}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  💾 Scatter plot saved → {out_path}")

    # ──────────────────────────────────────────
    # 4. Summary bar: ρ across all models
    # ──────────────────────────────────────────
    if len(summary_rows) > 1:
        sdf = pd.DataFrame(summary_rows).sort_values("rho")
        colors = ["#e15759" if row["p_value"] > 0.05 else ("#4e79a7" if row["rho"] < 0 else "#59a14f")
                  for _, row in sdf.iterrows()]

        fig, ax = plt.subplots(figsize=(10, max(5, len(sdf) * 0.55 + 1)))
        bars = ax.barh(sdf["model"], sdf["rho"], color=colors, edgecolor="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
        ax.axvline(0.5, color="#59a14f", linewidth=1, linestyle="--", alpha=0.6, label="Strong (ρ=0.5)")
        ax.axvline(-0.5, color="#e15759", linewidth=1, linestyle="--", alpha=0.6)

        for bar, (_, row) in zip(bars, sdf.iterrows()):
            sig_marker = "*" if row["p_value"] < 0.05 else "ns"
            ax.text(bar.get_width() + 0.01 if bar.get_width() >= 0 else bar.get_width() - 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['rho']:.2f} ({sig_marker})",
                    va="center", ha="left" if bar.get_width() >= 0 else "right",
                    fontsize=9.5)

        ax.set_title("Spearman ρ: Visual Salience → Bias Rate per Model", fontsize=14, fontweight="bold")
        ax.set_xlabel("Spearman ρ", fontsize=11)
        ax.set_xlim(-1.15, 1.15)
        ax.legend(fontsize=9, frameon=False)
        plt.tight_layout()

        summary_path = os.path.join(OUTPUT_DIR, "salience_rho_summary.png")
        plt.savefig(summary_path, dpi=300)
        plt.close()
        print(f"\n💾 Summary bar chart saved → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salience–Bias Spearman analysis")
    parser.add_argument("--model", default=None, help="Filter by model name substring (default: all models)")
    args = parser.parse_args()

    df = load_all_results()
    run_salience_analysis(df, model_filter=args.model)
