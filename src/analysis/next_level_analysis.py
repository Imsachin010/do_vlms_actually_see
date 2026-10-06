import pandas as pd

def compute_smoothed_bias(df, group_cols):
    agg = df.groupby(group_cols).agg({'is_biased': 'mean', 'error': 'mean'}).reset_index()
    max_err = agg['error'].max() if agg['error'].max() > 0 else 1
    agg['bias_rate'] = 0.7 * agg['is_biased'] + 0.3 * (agg['error'] / max_err)
    return agg

def run():
    import os
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
                if "predicted" in rdf.columns and "true_legs" in rdf.columns:
                    rdf["error"] = (rdf["predicted"] - rdf["true_legs"]).abs()
                all_results.append(rdf)
    
    df_raw = pd.concat(all_results, ignore_index=True)
    df_raw = df_raw.dropna(subset=["is_biased", "concept", "background", "model"])
    
    prior = pd.read_csv("data/processed/prior_strength_scores.csv")
    prior["concept"] = prior["concept"].str.lower()
    
    # 1. Global Aggregation by Background only
    print("=== Overall Bias Split By Background ===")
    combined_bg = compute_smoothed_bias(df_raw, ["model", "background"])
    print(combined_bg[["model", "background", "is_biased", "error", "bias_rate"]].sort_values(["model", "background"]).to_string(index=False))
    
    # 2. Correlation isolated by background for each model
    models = sorted(df_raw["model"].unique())
    for mod in models:
        print(f"\n=== Correlation Split By Background ({mod}) ===")
        mod_df = df_raw[df_raw["model"] == mod]
        mod_concept_bg = compute_smoothed_bias(mod_df, ["concept", "background"])
        mod_concept_bg["concept"] = mod_concept_bg["concept"].str.lower()
        
        merged = prior.merge(mod_concept_bg, on="concept", suffixes=("_prior", "_new"))
        if "bias_rate_new" in merged.columns:
            merged["bias_rate"] = merged["bias_rate_new"]
        
        for bg in sorted(merged["background"].unique()):
            subset = merged[merged["background"] == bg]
            if len(subset) > 1:
                corr = subset["prior_strength"].corr(subset["bias_rate"], method="spearman")
                print(f"Background: {bg:7} | Spearman rank correlation: {corr:8.4f}")
            else:
                print(f"Background: {bg:7} | Spearman rank correlation: Insufficient data")

if __name__ == "__main__":
    run()
