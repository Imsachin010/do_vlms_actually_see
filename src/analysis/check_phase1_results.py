import pandas as pd
from pathlib import Path


def load_data():
    wiki = pd.read_csv("data/processed/wikipedia_scores.csv")
    clip = pd.read_csv("data/processed/clip_scores.csv")
    model = pd.read_csv("data/processed/model_confidence_scores.csv")

    # Normalize concept names
    wiki["concept"] = wiki["concept"].str.lower()
    clip["concept"] = clip["concept"].str.lower()
    model["concept"] = model["concept"].str.lower()

    return wiki, clip, model


def merge_data(wiki, clip, model):
    df = wiki.merge(clip, on="concept").merge(model, on="concept")

    print("\nMerged Data Sample:")
    print(df[["concept", "pageviews_norm", "clip_zscore", "model_confidence"]].head())

    return df


def compute_correlations(df):
    results = {}

    results["wiki_vs_clip"] = df["pageviews_norm"].corr(
        df["clip_zscore"], method="spearman"
    )

    results["wiki_vs_model"] = df["pageviews_norm"].corr(
        df["model_confidence"], method="spearman"
    )

    results["clip_vs_model"] = df["clip_zscore"].corr(
        df["model_confidence"], method="spearman"
    )

    return results


def print_results(results):
    print("\n=== Correlation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Interpretation ===")

    if results["wiki_vs_model"] > 0.6:
        print("STRONG signal → NeurIPS-level potential")
    elif results["wiki_vs_model"] > 0.4:
        print("MEDIUM signal → TMLR-level")
    else:
        print("WEAK signal → Need improvement / rely on Phase 3")


def save_results(df, results):
    Path("results").mkdir(exist_ok=True)

    df.to_csv("results/phase1_merged.csv", index=False)

    with open("results/phase1_summary.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    print("\nSaved results to /results/")


def run():
    wiki, clip, model = load_data()
    df = merge_data(wiki, clip, model)

    results = compute_correlations(df)

    print_results(results)
    save_results(df, results)


if __name__ == "__main__":
    run()