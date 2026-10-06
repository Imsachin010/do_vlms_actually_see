import pandas as pd
from pathlib import Path


def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)


def run():
    wiki = pd.read_csv("data/processed/wikipedia_scores.csv")
    clip = pd.read_csv("data/processed/clip_scores.csv")
    model = pd.read_csv("data/processed/model_confidence_scores.csv")

    # normalize names
    wiki["concept"] = wiki["concept"].str.lower()
    clip["concept"] = clip["concept"].str.lower()
    model["concept"] = model["concept"].str.lower()

    df = wiki.merge(clip, on="concept").merge(model, on="concept")

    # normalize all proxies to same scale
    df["wiki_norm"] = normalize(df["pageviews_norm"])
    df["clip_norm_final"] = normalize(df["clip_zscore"])
    df["model_norm"] = normalize(df["model_confidence"])

    # weighted combination (IMPORTANT)
    df["prior_strength"] = (
        0.4 * df["wiki_norm"] +
        0.2 * df["clip_norm_final"] +
        0.4 * df["model_norm"]
    )

    # Save
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/prior_strength_scores.csv", index=False)

    print("\nFinal Prior Strength:")
    print(df[["concept", "prior_strength"]].sort_values(by="prior_strength", ascending=False))


if __name__ == "__main__":
    run()