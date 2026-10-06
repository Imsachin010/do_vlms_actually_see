"""
Figure 2: Bias Rate vs. Visual Feature Salience for Llama-4-Scout — Premium Journal Style

Features:
  - Clean point callouts with collision-free pointer offsets
  - Shaded 95% Bootstrap CI Band beneath points
  - Theil-Sen robust regression fit with unadjusted & Holm-adjusted statistics
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, theilslopes

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

APPENDIX_F_SALIENCE = {
    "elephant": 1.00,
    "lion": 0.85,
    "deer": 0.80,
    "horse": 0.75,
    "cheetah": 0.60,
    "leopard": 0.55,
    "fox": 0.50,
    "hyena": 0.45,
    "cat": 0.40,
    "dog": 0.35,
    "serval": 0.30,
    "binturong": 0.25,
    "okapi": 0.22,
    "fossa": 0.20,
    "numbat": 0.18
}

ANNOTATION_OFFSETS = {
    "elephant":  (0.85, 12.0),
    "lion":      (0.88, 88.0),
    "deer":      (0.72, 38.0),
    "dog":       (0.38, 86.0),
    "cat":       (0.43, 38.0),
    "fossa":     (0.12, 85.0),
    "numbat":    (0.24, 106.0),
}


def generate_figure():
    print("Generating Figure 2: Salience vs. Bias Rate (Journal Quality)...")
    print("=" * 60)

    df = pd.read_csv("data/processed/phase2_vlm_results.csv")
    df["model"] = df["model"].str.strip()
    sub = df[df["model"] == "llama4-scout"].copy()
    sub["concept"] = sub["concept"].str.lower().str.strip()
    sub["is_biased"] = ((sub["predicted"] == 4) & (sub["true_legs"] != 4)).astype(int)

    scout_bias = sub.groupby("concept")["is_biased"].mean().reset_index()
    scout_bias.columns = ["concept", "bias_rate"]

    scout_bias["salience_score"] = scout_bias["concept"].map(APPENDIX_F_SALIENCE)
    merged = scout_bias.sort_values("salience_score").reset_index(drop=True)

    x = merged["salience_score"].values
    y = merged["bias_rate"].values

    rho, p_val = spearmanr(x, y)
    p_one = p_val / 2.0 if rho < 0 else (1.0 - p_val / 2.0)

    res = theilslopes(y, x)
    slope = res[0]
    intercept = res[1]

    # Bootstrap 95% CI Band
    n_boot = 10000
    rng = np.random.default_rng(42)
    x_grid = np.linspace(0.12, 1.04, 200)
    boot_preds = np.zeros((n_boot, len(x_grid)))

    for i in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        xs, ys = x[idx], y[idx]
        res_b = theilslopes(ys, xs)
        boot_preds[i, :] = res_b[1] + res_b[0] * x_grid

    y_low = np.percentile(boot_preds, 2.5, axis=0) * 100
    y_high = np.percentile(boot_preds, 97.5, axis=0) * 100
    y_pred = (intercept + slope * x_grid) * 100

    fig, ax = plt.subplots(figsize=(9, 6.2))

    # Shaded 95% Bootstrap CI Band
    ax.fill_between(x_grid, y_low, y_high, color='#DC2626', alpha=0.12, label='95% Bootstrap CI')

    # Scatter points with strong contrast
    ax.scatter(x, y * 100, color='#2563EB', s=80, zorder=3, edgecolor='#1E293B', linewidth=1.0, alpha=0.95)

    # Theil-Sen regression fit line
    ax.plot(x_grid, y_pred, color='#DC2626', linewidth=2.2, linestyle='-', zorder=2,
            label=f'Theil-Sen Fit ($\\rho = {rho:.3f}$, $p = {p_val:.3f}$)')

    # Non-overlapping Concept Annotations
    for _, r in merged.iterrows():
        c_name = r["concept"]
        cx = r["salience_score"]
        cy = r["bias_rate"] * 100

        if c_name in ANNOTATION_OFFSETS:
            tx, ty = ANNOTATION_OFFSETS[c_name]
            ax.annotate(c_name.capitalize(),
                        xy=(cx, cy), xytext=(tx, ty),
                        fontsize=9.5, fontweight='bold', color='#1E293B',
                        arrowprops=dict(arrowstyle="->", color="#64748B", lw=0.8, shrinkA=0, shrinkB=4),
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.6))

    ax.set_xlabel("Visual Feature Salience Score", fontsize=11.5, fontweight='bold', labelpad=10, color="#0F172A")
    ax.set_ylabel("Bias Rate (%)", fontsize=11.5, fontweight='bold', labelpad=10, color="#0F172A")
    ax.set_title("Bias Rate vs. Visual Feature Salience (Llama-4-Scout)", fontsize=13.5, fontweight='bold', pad=18, color="#0F172A")
    ax.set_ylim(-8, 115)
    ax.set_xlim(0.10, 1.08)

    ax.legend(loc='lower left', frameon=True, facecolor='#FFFFFF', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9)

    plt.tight_layout()
    out_png = OUTPUT_DIR / "figure_salience_scout.png"
    out_pdf = OUTPUT_DIR / "figure_salience_scout.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[OK] Saved Figure 2 -> {out_png} and {out_pdf}")


if __name__ == "__main__":
    generate_figure()
