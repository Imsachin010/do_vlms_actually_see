"""
Figure 5: Background Context Effect on Prior-Aligned Bias Rate — Clean Journal Style

Compares mean per-concept bias rate (%) on Plain Background (Blue) vs. Natural Context (Amber)
across counterfactual vision models. Standard errors (SE) computed over N=15 concepts per model.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.titlesize'] = 13.5
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

DISPLAY_NAMES = {
    "Gemma-3-4B":            "Gemma-3-4B (4B)",
    "InstructBLIP-FlanT5XL": "InstructBLIP-FlanT5-XL (4B)",
    "LLaVA-v1.6-Mistral-7B": "LLaVA-v1.6 (7B)",
    "MiniCPM-V":             "MiniCPM-V (8B)",
    "Llama-3.2-V":           "Llama-3.2-Vision (11B)",
    "SmolVLM2-2.2B":         "SmolVLM2 (2.2B)",
    "Qwen2-VL-2B":           "Qwen2-VL (2B)",
    "Qwen2.5-VL-7B":         "Qwen2.5-VL (7B)",
    "LLaVA-1.5":             "LLaVA-1.5 (7B)",
    "Gemma-4-31B":           "Gemma-4-31B (31B)",
    "Llama-4-Scout":         "Llama-4-Scout (17B)",
    "Qwen3-VL-235B-A22B":    "Qwen3-VL-235B (235B)",
}


def get_background_effects():
    records = []

    def process_df(df, model_name, pred_col="predicted", legs_col="true_legs", bg_col="background", concept_col="concept"):
        df["_pred"] = pd.to_numeric(df[pred_col], errors="coerce")
        df["_legs"] = pd.to_numeric(df[legs_col], errors="coerce")
        # Counterfactual non-4 leg queries: bias is predicting 4
        cf = df[df["_legs"] != 4].copy()
        cf["_is_biased"] = (cf["_pred"] == 4).astype(int)
        cf["_bg"] = cf[bg_col].str.lower().str.strip()
        cf["_concept"] = cf[concept_col].str.lower().str.strip()

        for bg in ["plain", "natural"]:
            sub = cf[cf["_bg"].str.contains(bg)]
            pc = sub.groupby("_concept")["_is_biased"].mean() * 100.0
            mean_bias = float(pc.mean()) if len(pc) > 0 else 0.0
            se = float(pc.std() / (len(pc) ** 0.5)) if len(pc) > 1 else 0.0
            records.append({
                "model": model_name,
                "background": bg.capitalize(),
                "bias_rate": mean_bias,
                "se": se,
                "n_concepts": len(pc)
            })

    # Load Scout
    if Path("data/processed/phase2_vlm_results.csv").exists():
        df = pd.read_csv("data/processed/phase2_vlm_results.csv")
        sub = df[df["model"].str.strip() == "llama4-scout"]
        process_df(sub, "Llama-4-Scout")

    # Load LLaVA-1.5
    if Path("data/processed/llava_results.csv").exists():
        df = pd.read_csv("data/processed/llava_results.csv")
        process_df(df, "LLaVA-1.5")

    # Load Gemma-4-31B
    if Path("data/raw/results_hf/gemma_4_31b_results.csv").exists():
        df = pd.read_csv("data/raw/results_hf/gemma_4_31b_results.csv")
        process_df(df, "Gemma-4-31B")

    # Load Qwen3-VL
    if Path("data/raw/results_hf/qwen3_vl_235b_a22b_results.csv").exists():
        df = pd.read_csv("data/raw/results_hf/qwen3_vl_235b_a22b_results.csv")
        process_df(df, "Qwen3-VL-235B-A22B")

    # Load 6 Models from results10 if present
    r10_map = {
        "smolvlm2_results.csv": "SmolVLM2-2.2B",
        "gemma3_4b_results.csv": "Gemma-3-4B",
        "instructblip_results.csv": "InstructBLIP-FlanT5XL",
        "qwen2vl_2b_results.csv": "Qwen2-VL-2B",
        "qwen25_vl_results.csv": "Qwen2.5-VL-7B",
        "llava16_mistral_results.csv": "LLaVA-v1.6-Mistral-7B",
    }
    for fn, mn in r10_map.items():
        fp = os.path.join("Figures for Paper/results10", fn)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            process_df(df, mn)

    # Load Ollama models
    if Path("data/processed/phase3_ollama_results.csv").exists():
        df = pd.read_csv("data/processed/phase3_ollama_results.csv")
        for o_m, mn in [("minicpm-v", "MiniCPM-V"), ("llama3.2-vision", "Llama-3.2-V")]:
            sub = df[df["model"] == o_m].copy()
            if len(sub) > 0:
                process_df(sub, mn)

    return pd.DataFrame(records)


def generate_figure():
    print("Generating Figure 5: Background Effect (Clean Journal Style)...")
    print("=" * 60)

    bg_df = get_background_effects()

    plain = bg_df[bg_df["background"] == "Plain"].set_index("model")
    natural = bg_df[bg_df["background"] == "Natural"].set_index("model")

    # Sort models by Plain bias rate descending
    models = plain.sort_values("bias_rate", ascending=False).index.tolist()

    plain_rates = [plain.loc[m, "bias_rate"] for m in models]
    natural_rates = [natural.loc[m, "bias_rate"] for m in models]

    plain_se = [plain.loc[m, "se"] for m in models]
    natural_se = [natural.loc[m, "se"] for m in models]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6.5))

    rects1 = ax.bar(
        x - width / 2,
        plain_rates,
        width,
        label="Plain Background",
        color="#2563EB",
        yerr=plain_se,
        capsize=4,
        edgecolor="#1E293B",
        linewidth=0.8,
    )

    rects2 = ax.bar(
        x + width / 2,
        natural_rates,
        width,
        label="Natural Context",
        color="#D97706",
        yerr=natural_se,
        capsize=4,
        edgecolor="#1E293B",
        linewidth=0.8,
    )

    ax.set_ylabel("Bias Rate (%)", fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_title(
        "Background Context Effect on Prior-Aligned Bias Rate",
        fontsize=13.5,
        fontweight="bold",
        pad=20,
        color="#0F172A",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(m, m) for m in models],
        fontsize=9.5,
        fontweight="normal",
        color="#1E293B",
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_ylim(0, 130)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.95,
        edgecolor="#CBD5E1",
        fontsize=9.5,
    )

    # Annotate Plain bars
    for idx, rect in enumerate(rects1):
        h = rect.get_height()
        top_y = h + plain_se[idx]
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, top_y + 2.0),
            ha="left",
            va="bottom",
            rotation=45,
            fontsize=8.0,
            fontweight="normal",
            color="#1E3A8A",
        )

    # Annotate Natural bars
    for idx, rect in enumerate(rects2):
        h = rect.get_height()
        top_y = h + natural_se[idx]
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, top_y + 2.0),
            ha="left",
            va="bottom",
            rotation=45,
            fontsize=8.0,
            fontweight="normal",
            color="#92400E",
        )

    plt.tight_layout()

    out_png = OUTPUT_DIR / "figure_background_effect.png"
    out_pdf = OUTPUT_DIR / "figure_background_effect.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved background effect figure -> {out_png} and {out_pdf}")


if __name__ == "__main__":
    generate_figure()
