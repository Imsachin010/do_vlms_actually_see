"""
concept_salience_analysis.py
─────────────────────────────
Implements a two-metric salience framework and correlates each 
against Llama-4-Scout's per-concept bias rate.

METRIC 1 — Concept Frequency Score
    Source: Wikipedia pageviews (monthly, 2020-2025), already fetched.
    Interpretation: How familiar/common a concept is in public knowledge.

METRIC 2 — CLIP Separability Score
    Definition: inter_class_distance / intra_class_variance
    Interpretation: How visually distinct a concept is from all others.
    High separability → model can anchor on the concept visually.
    Low separability → concept blurs into similar-looking animals.

Outputs saved to: results/salience_analysis/
"""

import os
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ───────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[2]
IMAGES_DIR  = BASE / "data" / "raw" / "canonical_images"
WIKI_CSV    = BASE / "data" / "processed" / "wikipedia_scores.csv"
PHASE2_CSV  = BASE / "data" / "processed" / "phase2_vlm_results.csv"
OUT_DIR     = BASE / "results" / "salience_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Concept Frequency Score (Wikipedia pageviews)
# ════════════════════════════════════════════════════════════════════════════

def load_frequency_scores() -> pd.DataFrame:
    wiki = pd.read_csv(WIKI_CSV)
    freq = wiki[["concept", "pageviews", "pageviews_norm"]].copy()
    freq.columns = ["concept", "pageviews_raw", "concept_frequency_score"]
    freq["concept"] = freq["concept"].str.lower()
    return freq


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: CLIP Separability Score = inter_dist / intra_var
# ════════════════════════════════════════════════════════════════════════════

def load_all_embeddings(model, preprocess):
    """Encode all canonical images per concept. Returns {concept: np.array shape (N, D)}."""
    concept_embeddings = {}
    for concept_dir in sorted(IMAGES_DIR.iterdir()):
        if not concept_dir.is_dir():
            continue
        concept = concept_dir.name.lower()
        embs = []
        for img_file in concept_dir.iterdir():
            if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".avif"}:
                try:
                    img = preprocess(Image.open(img_file).convert("RGB")).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        emb = model.encode_image(img)
                    embs.append(emb.cpu().numpy()[0])
                except Exception as e:
                    print(f"  ⚠ Skip {img_file.name}: {e}")
        if embs:
            concept_embeddings[concept] = np.stack(embs)
            print(f"  {concept:<14} {len(embs)} images encoded")
    return concept_embeddings


def compute_separability(concept_embeddings: dict) -> pd.DataFrame:
    """
    For each concept C:
      intra_var  = mean pairwise cosine distance within C (spread of own images)
      inter_dist = mean cosine distance from C's centroid to all other concepts' centroids
      separability = inter_dist / (intra_var + ε)
    """
    concepts = list(concept_embeddings.keys())
    centroids = {c: concept_embeddings[c].mean(axis=0) for c in concepts}
    centroid_matrix = np.stack([centroids[c] for c in concepts])  # (N_concepts, D)

    rows = []
    for concept in concepts:
        embs = concept_embeddings[concept]

        # Intra-class variance: mean pairwise cosine distance
        if len(embs) >= 2:
            dist_matrix = cosine_distances(embs)
            n = len(embs)
            intra_var = dist_matrix[np.triu_indices(n, k=1)].mean()
        else:
            intra_var = 1e-6  # single image — treat as no spread

        # Inter-class distance: cosine distance from this centroid to all others
        own_centroid = centroids[concept].reshape(1, -1)
        other_centroids = np.stack([centroids[c] for c in concepts if c != concept])
        inter_dist = cosine_distances(own_centroid, other_centroids).mean()

        separability = inter_dist / (intra_var + 1e-6)

        rows.append({
            "concept":         concept,
            "intra_var":       intra_var,
            "inter_dist":      inter_dist,
            "clip_separability_raw": separability,
        })

    df = pd.DataFrame(rows)
    # Normalise separability to [0, 1]
    mn, mx = df["clip_separability_raw"].min(), df["clip_separability_raw"].max()
    df["clip_separability_score"] = (df["clip_separability_raw"] - mn) / (mx - mn + 1e-8)
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: Correlation Analysis + Plots
# ════════════════════════════════════════════════════════════════════════════

def load_bias_rates() -> pd.DataFrame:
    df = pd.read_csv(PHASE2_CSV)
    scout = df[df["model"] == "llama4-scout"]
    bias = scout.groupby("concept")["is_biased"].mean().mul(100).reset_index()
    bias.columns = ["concept", "bias_rate"]
    bias["concept"] = bias["concept"].str.lower()
    return bias


def spearman_report(x_series, y_series, label):
    valid = pd.concat([x_series, y_series], axis=1).dropna()
    if len(valid) < 5:
        return None, None
    rho, p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    sig = "✅ STRONG" if (p < 0.05 and abs(rho) > 0.5) else \
          "⚠️  MEDIUM" if (p < 0.05 and abs(rho) > 0.3) else "❌ NULL"
    print(f"  {label:<35} ρ = {rho:+.3f}   p = {p:.4f}   {sig}")
    return rho, p


def scatter_with_regression(ax, x_vals, y_vals, labels, title, xlabel):
    color = "#4e79a7"
    ax.scatter(x_vals, y_vals, s=120, color=color, edgecolors="black", linewidths=0.6, zorder=3)
    z = np.polyfit(x_vals, y_vals, 1)
    xl = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xl, np.polyval(z, xl), "--", color="#888", linewidth=1.4, alpha=0.7)
    for x, y, lbl in zip(x_vals, y_vals, labels):
        ax.annotate(lbl.capitalize(), (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=8, color="#333")
    rho, p = spearmanr(x_vals, y_vals)
    sig = "*" if p < 0.05 else "ns"
    ax.set_title(f"{title}\nρ = {rho:.3f},  p = {p:.4f} ({sig})", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Bias Rate (%)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(-5, 112)
    ax.grid(True, linestyle="--", alpha=0.4)


def tier_bar_chart(merged, out_dir):
    """Bar chart: avg bias per tier for Concept Frequency + CLIP Separability."""
    wiki = pd.read_csv(WIKI_CSV)[["concept", "tier"]]
    wiki["concept"] = wiki["concept"].str.lower()
    merged2 = merged.merge(wiki, on="concept", how="left")

    tier_stats = merged2.groupby("tier")[["bias_rate", "concept_frequency_score",
                                          "clip_separability_score"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tier_stats))
    w = 0.25
    ax.bar(x - w, tier_stats["bias_rate"],                   w, label="Bias Rate (%)",                color="#e15759", edgecolor="black", linewidth=0.6)
    ax.bar(x,     tier_stats["concept_frequency_score"]*100, w, label="Concept Frequency Score (×100)", color="#4e79a7", edgecolor="black", linewidth=0.6)
    ax.bar(x + w, tier_stats["clip_separability_score"]*100, w, label="CLIP Separability Score (×100)", color="#59a14f", edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(["Tier 1\n(Common)", "Tier 2\n(Moderate)", "Tier 3\n(Obscure)"], fontsize=11)
    ax.set_ylabel("Score / %", fontsize=11)
    ax.set_title("Concept Frequency & CLIP Separability vs Bias Rate per Tier\n(Llama-4-Scout)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    out = out_dir / "tier_comparison_bar.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"\n💾 Tier bar chart → {out}")


def main():
    print("\n" + "="*65)
    print("STEP 1: Loading Concept Frequency Scores (Wikipedia pageviews)")
    print("="*65)
    freq_df = load_frequency_scores()
    print(freq_df[["concept", "concept_frequency_score"]].sort_values(
        "concept_frequency_score", ascending=False).to_string(index=False))

    print("\n" + "="*65)
    print("STEP 2: Computing CLIP Separability Scores")
    print("="*65)
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    concept_embeddings = load_all_embeddings(model, preprocess)
    sep_df = compute_separability(concept_embeddings)
    print("\n  Separability scores (high = visually distinct from others):")
    print(sep_df[["concept", "intra_var", "inter_dist", "clip_separability_score"]]
          .sort_values("clip_separability_score", ascending=False).to_string(index=False))

    print("\n" + "="*65)
    print("STEP 3: Correlation with Llama-4-Scout Bias Rate")
    print("="*65)
    bias_df  = load_bias_rates()
    merged   = bias_df.merge(freq_df, on="concept", how="left")
    merged   = merged.merge(sep_df[["concept", "clip_separability_score",
                                     "intra_var", "inter_dist"]], on="concept", how="left")

    print("\n  Bias vs Concept Frequency Score:")
    rho_freq, p_freq = spearman_report(merged["concept_frequency_score"],
                                        merged["bias_rate"],
                                        "Concept Frequency Score")

    print("\n  Bias vs CLIP Separability Score:")
    rho_sep, p_sep = spearman_report(merged["clip_separability_score"],
                                      merged["bias_rate"],
                                      "CLIP Separability Score")

    # ── Save merged CSV ──────────────────────────────────────────────────
    out_csv = OUT_DIR / "concept_salience_metrics.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\n💾 Metrics CSV → {out_csv}")

    # ── Save correlation summary ─────────────────────────────────────────
    corr_csv = OUT_DIR / "correlation_summary.csv"
    pd.DataFrame([
        {"metric": "Concept Frequency Score (Wikipedia pageviews)", "spearman_rho": rho_freq, "p_value": p_freq},
        {"metric": "CLIP Separability Score (inter/intra)",         "spearman_rho": rho_sep,  "p_value": p_sep},
    ]).to_csv(corr_csv, index=False)
    print(f"💾 Correlation summary → {corr_csv}")

    # ── Scatter plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_with_regression(axes[0], merged["concept_frequency_score"].values,
                             merged["bias_rate"].values, merged["concept"].values,
                             "Bias vs Concept Frequency Score", "Concept Frequency Score")
    scatter_with_regression(axes[1], merged["clip_separability_score"].values,
                             merged["bias_rate"].values, merged["concept"].values,
                             "Bias vs CLIP Separability Score", "CLIP Separability Score")
    plt.suptitle("Llama-4-Scout: Salience Metrics vs Bias Rate",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_scatter = OUT_DIR / "salience_vs_bias_scatter.png"
    plt.savefig(out_scatter, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Scatter plots → {out_scatter}")

    # ── Tier bar chart ────────────────────────────────────────────────────
    tier_bar_chart(merged, OUT_DIR)

    print("\n✅ All results saved to:", OUT_DIR)
    print("\nFull merged data:")
    print(merged[["concept", "bias_rate", "concept_frequency_score",
                  "clip_separability_score"]].sort_values("bias_rate").to_string(index=False))


if __name__ == "__main__":
    main()
