import requests
import pandas as pd
import yaml
import time
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger()

WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


def load_concepts(config_path="config/concepts.yaml"):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    concepts = []
    for tier, items in data["concepts"].items():
        for c in items:
            concepts.append((c, tier))

    return concepts

def fetch_pageviews(concept):
    concept_encoded = concept.replace(" ", "_")

    url = f"{WIKI_API}/en.wikipedia/all-access/all-agents/{concept_encoded}/monthly/20200101/20250101"

    headers = {
        "User-Agent": "VLM-Bias-Research/1.0 (your_email@example.com)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            logger.warning(f"{concept} → HTTP {response.status_code}")
            return 0

        data = response.json()

        if "items" not in data:
            logger.warning(f"No items for {concept}: {data}")
            return 0

        total_views = sum(item["views"] for item in data["items"])
        return total_views

    except Exception as e:
        logger.error(f"Error fetching {concept}: {e}")
        return 0



def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)


def run():
    logger.info("Starting Wikipedia Proxy...")

    concepts = load_concepts()

    results = []

    for concept, tier in concepts:
        logger.info(f"Fetching {concept}")
        views = fetch_pageviews(concept)

        results.append({
            "concept": concept,
            "tier": tier,
            "pageviews": views
        })

        time.sleep(0.5)  # avoid rate limit

    df = pd.DataFrame(results)

    # Normalize
    df["pageviews_norm"] = normalize(df["pageviews"])

    # Save raw
    Path("data/raw/wikipedia_pageviews").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/wikipedia_pageviews/pageviews.csv", index=False)

    # Save processed
    df.to_csv("data/processed/wikipedia_scores.csv", index=False)

    logger.info("Saved Wikipedia scores")

    # Validate tier ordering
    validate(df)


def validate(df):
    tier_means = df.groupby("tier")["pageviews_norm"].mean()

    logger.info(f"Tier means:\n{tier_means}")

    try:
        assert tier_means["tier1"] > tier_means["tier2"] > tier_means["tier3"]
        # logger.info("✅ Tier ordering valid")
    except AssertionError:
        logger.warning("⚠️ Tier ordering violated — revise concepts")


if __name__ == "__main__":
    run()