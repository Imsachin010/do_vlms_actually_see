import pandas as pd

def run():
    prior = pd.read_csv("data/processed/prior_strength_scores.csv")
    bias = pd.read_csv("data/processed/phase2_bias_per_concept.csv")

    prior["concept"] = prior["concept"].str.lower()
    bias["concept"] = bias["concept"].str.lower()

    # Use explicit suffixes to avoid column name collision if prior already has bias_rate
    df = prior.merge(bias, on="concept", suffixes=("_prior", "_phase2"))

    # Pick the bias_rate from phase2; fall back to plain name if no collision occurred
    if "bias_rate_phase2" in df.columns:
        df["bias_rate"] = df["bias_rate_phase2"]
    # If no suffix was added, bias_rate is already present as-is

    corr = df["prior_strength"].corr(df["bias_rate"], method="spearman")

    print("\n=== FINAL RESULT ===")
    print("Spearman Correlation:", corr)

    print("\nSorted Data:")
    print(df.sort_values("prior_strength", ascending=False))


if __name__ == "__main__":
    run()