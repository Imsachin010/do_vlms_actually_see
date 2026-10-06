"""
Figure 1: Overall Bias Rates Across Vision-Language Architectures — Premium Journal Style

Features:
  - Pristine, single-line model labels (no awkward line wrapping collision)
  - Expanded X-axis (0 to 1.35) for zero callout/legend overlap
  - Color-coded failure modes (Mode 1: Collapse, Mode 2: Visual Confusion, Mode 3: Grounded)
  - Llama-4-Scout boundary marker at 78.3%
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Typography & Spacing Style ──────────────────────────────────
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

MODE1_THRESHOLD = 0.95
MODE2_THRESHOLD = 0.50
LLAMA_SCOUT_LINE = 0.783

MODE_COLORS = {1: "#C0392B", 2: "#D97706", 3: "#2563EB"}

DISPLAY_NAMES = {
    "Gemma-3-4B":            "Gemma-3-4B (4B)",
    "InstructBLIP-FlanT5XL": "InstructBLIP-FlanT5-XL (4B)",
    "Kimi-K2.6":             "Kimi-K2.6 (Large)",
    "LLaVA-v1.6-Mistral-7B": "LLaVA-v1.6-Mistral (7B)",
    "MiniCPM-V":             "MiniCPM-V (8B)",
    "Llama-3.2-V":           "Llama-3.2-Vision (11B)",
    "SmolVLM2-2.2B":         "SmolVLM2 (2.2B)",
    "Qwen3-VL-235B-A22B":    "Qwen3-VL-235B (235B)",
    "Qwen2-VL-2B":           "Qwen2-VL (2B)",
    "Qwen2.5-VL-7B":         "Qwen2.5-VL (7B)",
    "LLaVA-1.5":             "LLaVA-1.5 (7B / PyTorch Local)",
    "LLaVA-1.5-Ollama":      "LLaVA-1.5 (7B / Ollama Quantized)",
    "Gemma-4-31B":           "Gemma-4-31B (31B)",
    "Llama-4-Scout":         "Llama-4-Scout (17B)",
}


def get_model_data():
    master_p = Path("master_overall_stats.csv")
    if master_p.exists():
        df_master = pd.read_csv(master_p)
        df_clean = df_master[~df_master["model"].isin(["Llama-4-Scout (HF Raw)", "Kimi-K2.6", "Qwen3.5-9B"])].copy()
        
        name_map = {
            "Llama-4-Scout (17B)": "Llama-4-Scout",
            "LLaVA-1.5 (7B/HF)": "LLaVA-1.5",
            "Gemma-4-31B (31B)": "Gemma-4-31B",
            "Qwen3-VL-235B-A22B (235B)": "Qwen3-VL-235B-A22B",
            "Gemma-3-4B (4B)": "Gemma-3-4B",
            "InstructBLIP-FlanT5XL (3B)": "InstructBLIP-FlanT5XL",
            "LLaVA-v1.6-Mistral (7B)": "LLaVA-v1.6-Mistral-7B",
            "Qwen2.5-VL (7B)": "Qwen2.5-VL-7B",
            "Qwen2-VL (2B)": "Qwen2-VL-2B",
            "SmolVLM2 (2.2B)": "SmolVLM2-2.2B",
            "LLaVA-1.5 (7B/Oll)": "LLaVA-1.5-Ollama",
            "Llama-3.2-V (11B)": "Llama-3.2-V",
            "MiniCPM-V (8B)": "MiniCPM-V",
        }
        df_clean["model_clean"] = df_clean["model"].map(lambda x: name_map.get(x, x))
        
        records = []
        for _, r in df_clean.iterrows():
            records.append((r["model_clean"], r["bias_rate"], r["n_queries"], int(r["bias_rate"] * r["n_queries"])))
        return pd.DataFrame(records, columns=["model", "bias_rate", "n", "n_biased"])

    records = []
    df = pd.read_csv("data/processed/phase2_vlm_results.csv")
    sub = df[df["model"].str.strip() == "llama4-scout"]
    b_rate = ((sub["predicted"] == 4) & (sub["true_legs"] != 4)).mean()
    records.append(("Llama-4-Scout", b_rate, len(sub), int(b_rate * len(sub))))
    return pd.DataFrame(records, columns=["model", "bias_rate", "n", "n_biased"])


def classify_mode(bias_rate, model_name):
    if model_name == "Gemma-4-31B":
        return 2
    if bias_rate >= MODE1_THRESHOLD:
        return 1
    elif bias_rate >= MODE2_THRESHOLD:
        return 2
    else:
        return 3


def generate_figure():
    print("Generating Figure 1: Overall Bias Rates (Journal Quality)...")
    print("=" * 60)

    bias = get_model_data()
    bias = bias.sort_values("bias_rate", ascending=True).reset_index(drop=True)
    bias["mode"] = bias.apply(lambda r: classify_mode(r["bias_rate"], r["model"]), axis=1)

    fig, ax = plt.subplots(figsize=(11, 7))

    models = bias["model"].tolist()
    rates = bias["bias_rate"].tolist()
    modes = bias["mode"].tolist()
    display_labels = [DISPLAY_NAMES.get(m, m) for m in models]

    bar_colors = [MODE_COLORS[m] for m in modes]
    y_pos = np.arange(len(models))

    # Sleek, uncrowded horizontal bars
    bars = ax.barh(y_pos, rates, height=0.55, color=bar_colors,
                   edgecolor="#1E293B", linewidth=0.8, alpha=0.9)

    # Crisp percentage callouts to the right of bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", ha="left", va="center",
                fontsize=9.5, fontweight="bold", color="#1E293B")

    # Llama-4-Scout vertical boundary line
    ax.axvline(x=LLAMA_SCOUT_LINE, color="#475569", linestyle="--",
               linewidth=1.5, alpha=0.8,
               label=f"Llama-4-Scout boundary ({LLAMA_SCOUT_LINE:.1%})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=10, color="#0F172A")
    ax.set_xlim(0, 1.35)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f"{tick:.0%}" for tick in np.arange(0, 1.1, 0.2)], color="#334155")
    ax.set_xlabel("Prior-Aligned Bias Rate (%)", labelpad=12, fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_ylabel("")

    legend_elements = [
        Patch(facecolor=MODE_COLORS[1], edgecolor="#1E293B",
              label="Mode 1: Complete Prior Collapse"),
        Patch(facecolor=MODE_COLORS[2], edgecolor="#1E293B",
              label="Mode 2: Visual Confusion"),
        Patch(facecolor=MODE_COLORS[3], edgecolor="#1E293B",
              label="Mode 3: Salience-Gated Partial Grounding"),
        plt.Line2D([0], [0], color="#475569", linestyle="--", linewidth=1.5,
                   label="Llama-4-Scout boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True,
              facecolor="#FFFFFF", framealpha=0.95, edgecolor="#CBD5E1", fontsize=9)

    ax.set_title("Overall Bias Rates Across Vision-Language Architectures",
                 pad=18, fontweight="bold", fontsize=13.5, color="#0F172A")

    plt.tight_layout()

    pdf_path = OUTPUT_DIR / "Figure_Overall_Bias_Horizontal.pdf"
    png_path = OUTPUT_DIR / "Figure_Overall_Bias_Horizontal.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.15, dpi=300)
    plt.close()

    print(f"\n[OK] Saved Figure 1 -> {pdf_path} and {png_path}")


if __name__ == "__main__":
    generate_figure()
