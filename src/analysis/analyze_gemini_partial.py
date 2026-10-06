import pandas as pd
from scipy.stats import spearmanr
import os

def run():
    path = "data/processed/gemini_results.csv"
    if not os.path.exists(path):
        print("No gemini results found.")
        return
        
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} successful Gemini predictions.")
    print("Counts by model:\n", df.groupby("model").size())
    
    agg = df.groupby(["model", "concept", "background"]).agg({"is_biased": "mean", "error": "mean"}).reset_index()
    max_err = agg["error"].max() if agg["error"].max() > 0 else 1
    agg["bias_rate"] = 0.7 * agg["is_biased"] + 0.3 * (agg["error"] / max_err)
    
    prior = pd.read_csv("data/processed/prior_strength_scores.csv")
    prior["concept"] = prior["concept"].str.lower()
    agg["concept"] = agg["concept"].str.lower()
    
    merged = agg.merge(prior, on="concept")
    
    for mod in merged["model"].unique():
        sub = merged[merged["model"] == mod]
        
        print(f"\n[{mod}] Partial Results (Concepts Covered: {sub['concept'].nunique()})")
        print(f"Overall average bias rate: {sub['bias_rate'].mean():.2%}")
        
        # Overall Correlation
        if len(sub) > 2:
            rho, p = spearmanr(sub["prior_strength"], sub["bias_rate"])
            print(f"Overall Correlation: rho = {rho:.3f}, p = {p:.4f}")
        
        # Split by Background
        for bg in ["plain", "natural"]:
            bg_sub = sub[sub["background"] == bg]
            print(f"  [{bg.upper()}] Average Bias: {bg_sub['bias_rate'].mean():.2%}")
            if len(bg_sub) >= 3:
                r, pv = spearmanr(bg_sub["prior_strength"], bg_sub["bias_rate"])
                print(f"  [{bg.upper()}] Correlation: rho = {r:.3f}, p = {pv:.4f} (n={len(bg_sub)} images)")
            else:
                print(f"  [{bg.upper()}] Correlation: insufficient data (n={len(bg_sub)})")

if __name__ == "__main__":
    run()
