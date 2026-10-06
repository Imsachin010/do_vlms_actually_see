"""
Figure: Perception Bypass Ablation — Comprehensive 14-Configuration Comparison

Compares accuracy on raw counterfactual images (Blue) vs. Scene Graph text grounding (Orange)
across all 14 evaluated model configurations (including FP16 vs Q4_0 precisions for Qwen2.5-VL and LLaVA-1.5).

Error bars represent 95% Wilson score confidence intervals computed over N=60 queries per model.
Text callouts are positioned cleanly above the error bar caps to guarantee zero overlap.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple:
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_correct / n_total
    denominator = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = (z / denominator) * math.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total
    )
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (lower, upper)


CONFIGS = [
    {"label": "Llama-4-Scout (17B)", "raw_c": 8, "raw_n": 60, "sg_c": 60, "sg_n": 60},
    {"label": "Gemma-4-31B (31B)", "raw_c": 7, "raw_n": 60, "sg_c": 60, "sg_n": 60},
    {"label": "Qwen3-VL-235B (235B)", "raw_c": 0, "raw_n": 60, "sg_c": 60, "sg_n": 60},
    {"label": "LLaVA-v1.6-Mistral (7B)", "raw_c": 0, "raw_n": 60, "sg_c": 60, "sg_n": 60},
    {"label": "InstructBLIP-FlanT5-XL (4B)", "raw_c": 0, "raw_n": 60, "sg_c": 60, "sg_n": 60},
    {"label": "Qwen2.5-VL (7B FP16)", "raw_c": 0, "raw_n": 60, "sg_c": 54, "sg_n": 60},
    {"label": "Qwen2.5-VL (7B Q4_0)", "raw_c": 0, "raw_n": 60, "sg_c": 50, "sg_n": 60},
    {"label": "LLaVA-1.5 (7B Q4_0)", "raw_c": 0, "raw_n": 60, "sg_c": 39, "sg_n": 60},
    {"label": "LLaVA-1.5 (7B FP16)", "raw_c": 0, "raw_n": 60, "sg_c": 30, "sg_n": 60},
    {"label": "MiniCPM-V (8B)", "raw_c": 0, "raw_n": 60, "sg_c": 30, "sg_n": 60},
    {"label": "Qwen2-VL (2B)", "raw_c": 0, "raw_n": 60, "sg_c": 30, "sg_n": 60},
    {"label": "SmolVLM2-2.2B (2.2B)", "raw_c": 0, "raw_n": 60, "sg_c": 28, "sg_n": 60},
    {"label": "Gemma-3-4B (4B)", "raw_c": 0, "raw_n": 60, "sg_c": 6, "sg_n": 60},
    {"label": "Llama-3.2-Vision (11B)", "raw_c": 0, "raw_n": 60, "sg_c": 0, "sg_n": 60},
]


def generate_figure():
    print("Generating Comprehensive Perception Bypass Ablation Figure (14 Configurations)...")
    print("=" * 70)

    labels = [c["label"] for c in CONFIGS]
    n_models = len(CONFIGS)

    raw_accs = [c["raw_c"] / c["raw_n"] for c in CONFIGS]
    sg_accs = [c["sg_c"] / c["sg_n"] for c in CONFIGS]

    raw_cis = [wilson_ci(c["raw_c"], c["raw_n"]) for c in CONFIGS]
    sg_cis = [wilson_ci(c["sg_c"], c["sg_n"]) for c in CONFIGS]

    raw_yerr = [
        [max(0.0, acc - ci[0]) for acc, ci in zip(raw_accs, raw_cis)],
        [max(0.0, ci[1] - acc) for acc, ci in zip(raw_accs, raw_cis)],
    ]
    sg_yerr = [
        [max(0.0, acc - ci[0]) for acc, ci in zip(sg_accs, sg_cis)],
        [max(0.0, ci[1] - acc) for acc, ci in zip(sg_accs, sg_cis)],
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n_models)
    width = 0.38

    rects1 = ax.bar(
        x - width / 2,
        [a * 100 for a in raw_accs],
        width,
        label="Raw Counterfactual Image Accuracy",
        color="#1F77B4",
        yerr=[[e * 100 for e in raw_yerr[0]], [e * 100 for e in raw_yerr[1]]],
        capsize=4,
        edgecolor="black",
        linewidth=0.8,
    )

    rects2 = ax.bar(
        x + width / 2,
        [a * 100 for a in sg_accs],
        width,
        label="Scene Graph Text Grounding Accuracy",
        color="#FF7F0E",
        yerr=[[e * 100 for e in sg_yerr[0]], [e * 100 for e in sg_yerr[1]]],
        capsize=4,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Perception Bypass Ablation: Raw Image vs. Scene Graph Text Grounding (N=60)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 130)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right")

    # Annotate Raw Vision bars above upper Wilson CI cap
    for idx, rect in enumerate(rects1):
        height = rect.get_height()
        upper_cap = raw_cis[idx][1] * 100
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, upper_cap + 2.5),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.annotate(
                "0%",
                xy=(rect.get_x() + rect.get_width() / 2, upper_cap + 1.5),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1F77B4",
            )

    # Annotate Scene Graph bars above upper Wilson CI cap
    for idx, rect in enumerate(rects2):
        height = rect.get_height()
        upper_cap = sg_cis[idx][1] * 100
        ax.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, upper_cap + 2.5),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    out_png = OUTPUT_DIR / "figure_perception_ablation.png"
    out_pdf = OUTPUT_DIR / "figure_perception_ablation.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved perception ablation figure -> {out_png} and {out_pdf}")


if __name__ == "__main__":
    generate_figure()
