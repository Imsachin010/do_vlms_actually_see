import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os

# Set publication style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def run():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Load Data
    all_results = []
    
    # Load original model results
    if os.path.exists("data/processed/phase2_vlm_results.csv"):
        llama = pd.read_csv("data/processed/phase2_vlm_results.csv")
        llama["model"] = "Llama-4-Scout"
        all_results.append(llama)
    
    if os.path.exists("data/processed/llava_results.csv"):
        llava = pd.read_csv("data/processed/llava_results.csv")
        llava["model"] = "LLaVA-1.5"
        all_results.append(llava)
        
    # Load new models from results10
    new_results_dir = "Figures for Paper/results10"
    if os.path.exists(new_results_dir):
        for f in os.listdir(new_results_dir):
            if f.endswith(".csv"):
                rdf = pd.read_csv(os.path.join(new_results_dir, f))
                # Normalize columns
                if "predicted" in rdf.columns and "true_legs" in rdf.columns:
                    rdf["error"] = (rdf["predicted"] - rdf["true_legs"]).abs()
                all_results.append(rdf)
    
    df_raw = pd.concat(all_results, ignore_index=True)
    df_raw = df_raw.dropna(subset=["is_biased", "concept", "background", "model"])
    
    def smooth_bias(df):
        agg = df.groupby(["model", "concept", "background"]).agg({
            "is_biased": "mean", 
            "error": "mean" if "error" in df.columns else "is_biased"
        }).reset_index()
        
        max_err = agg["error"].max() if agg["error"].max() > 0 else 1
        agg["bias_rate"] = 0.7 * agg["is_biased"] + 0.3 * (agg["error"] / max_err)
        return agg

    df_agg = smooth_bias(df_raw)
    
    prior = pd.read_csv("data/processed/prior_strength_scores.csv")
    prior["concept"] = prior["concept"].str.lower()
    df_agg["concept"] = df_agg["concept"].str.lower()
    
    df = df_agg.merge(prior, on="concept", suffixes=("_new", "_prior"))
    if "bias_rate_new" in df.columns:
        df["bias_rate"] = df["bias_rate_new"]
    
    df.to_csv("results/tables/vlm_test_results.csv", index=False)
    
    # Analysis 1: ANOVA 
    print("=== Analysis 1: Two-way ANOVA ===")
    model = ols('bias_rate ~ C(model) * C(background)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    anova_table.to_csv("results/tables/main_results_anova.csv")
    print(anova_table)
    
    # Analysis 2: Concept-Level Breakdown (Summary across all models)
    print("\n=== Analysis 2: Concept Breakdown ===")
    concept_perf = df.groupby('concept')['bias_rate'].mean().sort_values(ascending=False)
    concept_perf.to_csv("results/tables/concept_breakdown.csv")
    print("Concepts with HIGHEST overall bias rate across all models:")
    print(concept_perf.head())
    
    # Analysis 3: BDI (Background Dependence Index)
    print("\n=== Analysis 3: Background Dependence Index ===")
    def calc_bdi(x):
        nat = x[x['background'] == 'natural']
        pla = x[x['background'] == 'plain']
        
        if len(nat) == 0 or len(pla) == 0:
            return 0
        return nat['bias_rate'].values[0] - pla['bias_rate'].values[0]
    
    bdi = df.groupby(['model', 'concept']).apply(calc_bdi).rename("BDI").reset_index()
    merged_bdi = bdi.merge(prior, on='concept')
    merged_bdi.to_csv("results/tables/bdi_analysis.csv", index=False)
    
    for mod in merged_bdi["model"].unique():
        sub = merged_bdi[merged_bdi["model"] == mod]
        rho, p = spearmanr(sub['BDI'], sub['prior_strength'])
        print(f"{mod} BDI <-> Prior Strength: rho = {rho:.3f}, p = {p:.4f}")
        
    print("\nGenerating Figures...")
    
    # Figure 3: Background Effect Visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="model", y="bias_rate", hue="background", capsize=.1)
    plt.title("Background Effect on Bias Rate across Models")
    plt.ylabel("Bias Rate")
    plt.xticks(rotation=45, ha='right')
    plt.savefig("results/figures/figure3_background_effect.png", bbox_inches='tight')
    plt.close()

    # Figure 2: N-Panel Scatter Plot (Grid Layout)
    models_to_plot = sorted(df["model"].unique())
    n_models = len(models_to_plot)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, mod in enumerate(models_to_plot):
        ax = axes[idx]
        sub = df[df["model"]==mod]
        sns.scatterplot(data=sub, x="prior_strength", y="bias_rate", hue="background", style="background", ax=ax, alpha=0.7)
        ax.set_title(mod)
        if idx != 0:
            ax.get_legend().remove()
    
    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    fig.text(0.5, 0.04, 'Prior Strength (Conceptual Salience)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Observed Bias Rate', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig("results/figures/figure2_panels.pdf", bbox_inches='tight', format='pdf', dpi=300)
    plt.savefig("results/figures/figure2_panels.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Done generating figures.")

if __name__ == "__main__":
    run()
