"""
Figure 4: Audited 4-Category Stacked Error Distribution Across Vision-Language Architectures

Legend placed in bottom-right corner (loc="lower right").
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

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

COLORS = {
    "prior-aligned":     "#C0392B",  # red
    "off-by-one":        "#D4A574",  # tan
    "visual-confusion":  "#7F8C8D",  # grey
    "other":             "#FFFFFF",  # white with black border
}
COLOR_ORDER = ["prior-aligned", "off-by-one", "visual-confusion", "other"]

DISPLAY_NAMES = {
    "Gemma-3-4B (4B)":           "Gemma-3-4B (4B)",
    "InstructBLIP-FlanT5XL (3B)": "InstructBLIP-FlanT5-XL (4B)",
    "LLaVA-v1.6-Mistral (7B)":   "LLaVA-v1.6-Mistral (7B)",
    "MiniCPM-V (8B)":            "MiniCPM-V (8B)",
    "Llama-3.2-Vision (11B)":    "Llama-3.2-Vision (11B)",
    "SmolVLM2-2.2B (2.2B)":      "SmolVLM2 (2.2B)",
    "Qwen3-VL-235B (235B)":      "Qwen3-VL-235B (235B)",
    "Qwen2-VL-2B (2B)":          "Qwen2-VL (2B)",
    "Qwen2.5-VL-7B (7B)":        "Qwen2.5-VL (7B)",
    "LLaVA-1.5 (7B/HF)":         "LLaVA-1.5 (7B/HF)",
    "LLaVA-1.5 (7B/Oll)":        "LLaVA-1.5 (7B/Oll)",
    "Gemma-4-31B (31B)":         "Gemma-4-31B (31B)",
    "Llama-4-Scout (17B)":       "Llama-4-Scout (17B)",
}

CORE_MODELS_CONFIG = [
    ("Llama-4-Scout (17B)", "data/processed/phase2_vlm_results.csv", "predicted", "model", "llama4-scout"),
    ("Gemma-4-31B (31B)", "data/raw/results_hf/gemma_4_31b_results.csv", "predicted", None, None),
    ("LLaVA-1.5 (7B/HF)", "data/processed/llava_results.csv", "predicted", None, None),
    ("Qwen3-VL-235B (235B)", "data/raw/results_hf/qwen3_vl_235b_a22b_results.csv", "predicted", None, None),
    ("Gemma-3-4B (4B)", "Figures for Paper/results10/gemma3_4b_results.csv", "predicted", None, None),
    ("InstructBLIP-FlanT5XL (3B)", "Figures for Paper/results10/instructblip_results.csv", "predicted", None, None),
    ("LLaVA-v1.6-Mistral (7B)", "Figures for Paper/results10/llava16_mistral_results.csv", "predicted", None, None),
    ("Qwen2.5-VL-7B (7B)", "Figures for Paper/results10/qwen25_vl_results.csv", "predicted", None, None),
    ("Qwen2-VL-2B (2B)", "Figures for Paper/results10/qwen2vl_2b_results.csv", "predicted", None, None),
    ("SmolVLM2-2.2B (2.2B)", "Figures for Paper/results10/smolvlm2_results.csv", "predicted", None, None),
    ("LLaVA-1.5 (7B/Oll)", "data/processed/phase3_ollama_results.csv", "predicted", "model", "llava:7b"),
    ("Llama-3.2-Vision (11B)", "data/processed/phase3_ollama_results.csv", "predicted", "model", "llama3.2-vision"),
    ("MiniCPM-V (8B)", "data/processed/phase3_ollama_results.csv", "predicted", "model", "minicpm-v"),
]


def load_audited_4cat_data():
    rows = []

    for label, path, pred_col, filter_col, filter_val in CORE_MODELS_CONFIG:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)

        if filter_col and filter_val:
            df = df[df[filter_col] == filter_val].copy()

        c_col = "concept" if "concept" in df.columns else None
        l_col = "true_legs" if "true_legs" in df.columns else "legs"
        bg_col = "background" if "background" in df.columns else "bg"

        if bg_col not in df.columns and "image" in df.columns:
            df["background"] = df["image"].apply(lambda x: "plain" if "plain" in str(x) else "natural")
            bg_col = "background"

        df["_c"] = df[c_col].str.lower().str.strip()
        df["_l"] = df[l_col].astype(int)
        df["_b"] = df[bg_col].str.lower().str.strip()

        dedup = df.drop_duplicates(subset=["_c", "_l", "_b"], keep="last").copy()
        dedup["parsed_answer"] = pd.to_numeric(dedup[pred_col], errors="coerce")

        dedup["correct"] = (dedup["parsed_answer"] == dedup["_l"]).astype(int)
        errors = dedup[dedup["correct"] == 0].copy()
        n_errors = len(errors)

        if n_errors > 0:
            pa_n = int(((errors["parsed_answer"] == 4)).sum())
            
            obo_mask = (errors["parsed_answer"] != 4) & (abs(errors["parsed_answer"] - errors["_l"]) == 1)
            obo_n = int(obo_mask.sum())

            vis_mask = (errors["parsed_answer"] != 4) & (~obo_mask) & (errors["parsed_answer"].notna())
            vis_n = int(vis_mask.sum())

            oth_n = int(errors["parsed_answer"].isna().sum())

            pa_pct = pa_n / n_errors
            obo_pct = obo_n / n_errors
            vis_pct = vis_n / n_errors
            oth_pct = oth_n / n_errors
        else:
            pa_pct, obo_pct, vis_pct, oth_pct = 0.0, 0.0, 0.0, 0.0

        rows.append({
            "model": label,
            "prior-aligned": pa_pct,
            "off-by-one": obo_pct,
            "visual-confusion": vis_pct,
            "other": oth_pct,
            "total_errors": n_errors
        })

    return pd.DataFrame(rows)


def generate_figure():
    print("Generating Figure 4: Audited Stacked Error Distribution (Bottom-Right Legend)...")
    print("=" * 60)

    df_err = load_audited_4cat_data()
    df_err = df_err.sort_values("prior-aligned", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    models = df_err["model"].tolist()
    y_pos = np.arange(len(models))

    bottoms = np.zeros(len(models))

    for cat in COLOR_ORDER:
        vals = df_err[cat].values
        ax.barh(y_pos, vals, left=bottoms, height=0.60, label=cat,
                color=COLORS[cat], edgecolor="#1E293B", linewidth=0.8)
        bottoms += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in models], fontsize=9.5, fontweight="normal", color="#0F172A")
    ax.set_xlim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f"{tick:.0%}" for tick in np.arange(0, 1.1, 0.2)], color="#334155")
    ax.set_xlabel("Proportion of Total Errors", labelpad=12, fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_ylabel("")

    legend_elements = [
        Patch(facecolor=COLORS["prior-aligned"], edgecolor="#1E293B", label="Prior-aligned (pred=4)"),
        Patch(facecolor=COLORS["off-by-one"], edgecolor="#1E293B", label="Off-by-one (pred = 3 or 5)"),
        Patch(facecolor=COLORS["visual-confusion"], edgecolor="#1E293B", label="Visual confusion (non-canonical)"),
        Patch(facecolor=COLORS["other"], edgecolor="#1E293B", label="Other (API error / parse fail)"),
    ]
    
    # Legend placed in Bottom-Right Corner (loc="lower right")
    ax.legend(handles=legend_elements, loc="lower right",
              frameon=True, facecolor="#FFFFFF", framealpha=0.95, edgecolor="#CBD5E1", fontsize=9.5)

    ax.set_title("Error-Type Distribution Across Vision-Language Architectures",
                 pad=18, weight="bold", fontsize=13.5, color="#0F172A")

    plt.tight_layout()

    out_pdf1 = OUTPUT_DIR / "Figure_Error_Distribution_Stacked.pdf"
    out_png1 = OUTPUT_DIR / "Figure_Error_Distribution_Stacked.png"
    out_pdf2 = OUTPUT_DIR / "figure_error_distribution.pdf"
    out_png2 = OUTPUT_DIR / "figure_error_distribution.png"

    fig.savefig(out_pdf1, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_png1, bbox_inches="tight", pad_inches=0.15, dpi=300)
    fig.savefig(out_pdf2, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_png2, bbox_inches="tight", pad_inches=0.15, dpi=300)
    plt.close()

    print(f"\n[OK] Saved Figure 4 (Bottom-Right Legend) -> {out_pdf1} and {out_png1}")


if __name__ == "__main__":
    generate_figure()
