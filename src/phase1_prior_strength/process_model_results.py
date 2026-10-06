import pandas as pd
from pathlib import Path


def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)


def run():
    # Load raw results
    df = pd.read_csv("data/processed/model_raw_results.csv")

    print("\nRaw sample:")
    print(df.head())

    # Aggregate per concept (across models)
    concept_acc = df.groupby("concept").agg({
        "is_correct": "mean",
        "is_biased": "mean",
        "latency": "mean"
    }).reset_index()

    concept_acc.rename(columns={
        "is_correct": "accuracy",
        "is_biased": "bias_rate",
        "latency": "avg_latency"
    }, inplace=True)

    #
    # Normalize accuracy
    concept_acc["confidence_norm"] = normalize(concept_acc["accuracy"])

    # Bias score (lower bias = better)
    concept_acc["bias_score"] = 1 - concept_acc["bias_rate"]

    # Normalize latency (optional)
    concept_acc["latency_inv"] = 1 / (concept_acc["avg_latency"] + 1e-8)
    concept_acc["latency_norm"] = normalize(concept_acc["latency_inv"])

    # FINAL MODEL PROXY (core fix)
    concept_acc["model_confidence"] = (
        0.6 * concept_acc["confidence_norm"] +
        0.3 * concept_acc["bias_score"] +
        0.1 * concept_acc["latency_norm"]
    )

    # Save
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    concept_acc.to_csv("data/processed/model_confidence_scores.csv", index=False)

    print("\nProcessed Confidence Scores:")
    print(concept_acc.sort_values("model_confidence", ascending=False))


if __name__ == "__main__":
    run()