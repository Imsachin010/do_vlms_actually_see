import pandas as pd
import matplotlib.pyplot as plt
import os

def run():
    prior = pd.read_csv("data/processed/prior_strength_scores.csv")
    results = pd.read_csv("data/processed/phase2_vlm_results.csv")

    # Lowercase concept just in case
    prior["concept"] = prior["concept"].str.lower()
    results["concept"] = results["concept"].str.lower()

    # Merge
    df = results.merge(prior, on="concept")

    # Aggregate by concept + background
    agg = df.groupby(["concept", "background"]).agg({
        "is_biased": "mean",
        "prior_strength": "first"
    }).reset_index()

    print("\n=== Correlation by Background ===")

    for bg in ["plain", "natural"]:
        subset = agg[agg["background"] == bg]
        corr = subset["prior_strength"].corr(subset["is_biased"], method="spearman")
        print(f"{bg}: {corr:.4f}")

    print("\n=== Overall Correlation ===")
    corr_all = agg["prior_strength"].corr(agg["is_biased"], method="spearman")
    print(f"Overall: {corr_all:.4f}")

    # Plot logic
    os.makedirs("results", exist_ok=True)
    
    for bg in ["plain", "natural"]:
        subset = agg[agg["background"] == bg]

        plt.figure()
        plt.scatter(subset["prior_strength"], subset["is_biased"])

        for i, row in subset.iterrows():
            plt.text(row["prior_strength"], row["is_biased"], row["concept"])

        plt.xlabel("Prior Strength")
        plt.ylabel("Bias Rate")
        plt.title(f"Bias vs Prior ({bg})")

        out_path = f"results/bias_vs_prior_{bg}.png"
        plt.savefig(out_path)
        print(f"\nSaved plot to {out_path}")
        # Close the figure to free up memory 
        plt.close()

if __name__ == "__main__":
    run()
