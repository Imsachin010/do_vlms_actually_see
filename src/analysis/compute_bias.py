import pandas as pd

def run():
    df = pd.read_csv("data/processed/phase2_vlm_results.csv")

    concept_bias = df.groupby("concept").agg({
        "is_biased": "mean",
        "error": "mean"
    }).reset_index()

    # combine both
    concept_bias["bias_rate"] = (
        0.7 * concept_bias["is_biased"] +
        0.3 * (concept_bias["error"] / concept_bias["error"].max())
    )

    print("\nBias per Concept:")
    print(concept_bias.sort_values("bias_rate", ascending=False))

    concept_bias.to_csv("data/processed/phase2_bias_per_concept.csv", index=False)


if __name__ == "__main__":
    run()