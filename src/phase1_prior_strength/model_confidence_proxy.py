import os
import pandas as pd
import time
from pathlib import Path
from src.utils.logger import setup_logger
from dotenv import load_dotenv
from src.phase1_prior_strength.groq_client import GroqVLM

logger = setup_logger()
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

MODELS = {
    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    
}

PROMPT = "What animal is this? Answer with one word."

def clean_text(text):
    return text.lower().replace(".", "").strip()


def is_correct_prediction(concept, prediction):
    prediction = clean_text(prediction)

    synonyms = {
        "dog": ["dog"],
        "cat": ["cat"],
        "lion": ["lion"],
        "elephant": ["elephant"],
        "horse": ["horse"],
        "cheetah": ["cheetah"],
        "leopard": ["leopard"],
        "fox": ["fox"],
        "deer": ["deer"],
        "hyena": ["hyena"],
        "serval": ["serval"],
        "binturong": ["binturong", "bearcat"],
        "numbat": ["numbat"],
        "fossa": ["fossa"],
        "okapi": ["okapi"],
    }

    return int(prediction in synonyms.get(concept, [concept]))


def is_biased_prediction(concept, prediction):
    prediction = clean_text(prediction)

    # Generic collapse cases
    GENERIC = ["cat", "dog", "animal", "mammal"]

    # If rare animal predicted as generic/common → bias
    if concept in ["serval", "binturong", "numbat", "fossa", "okapi"]:
        if prediction in GENERIC:
            return 1

    return 0

def load_images(base_path="data/raw/canonical_images"):
    data = []

    for concept in os.listdir(base_path):
        concept_path = os.path.join(base_path, concept)

        if not os.path.isdir(concept_path):
            continue

        for file in os.listdir(concept_path):
            if file.endswith((".jpg", ".jpeg", ".png")):
                data.append({
                    "concept": concept.lower(),
                    "path": os.path.join(concept_path, file)
                })

    return data


def run():
    logger.info("Starting Model Confidence Proxy...")

    images = load_images()

    results = []

    for model_name, model_id in MODELS.items():
        logger.info(f"Running model: {model_name}")

        client = GroqVLM(API_KEY, model_id)

        for item in images:
            concept = item["concept"]
            path = item["path"]

            try:
                response, latency = client.query(path, PROMPT)

                if response is None:
                    continue

                prediction = clean_text(response)

                is_correct = is_correct_prediction(concept, prediction)
                is_biased = is_biased_prediction(concept, prediction)  

                results.append({
                    "model": model_name,
                    "concept": concept,
                    "image": path,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "is_biased": is_biased,
                    "latency": latency,
                })

                logger.info(f"{model_name} | {concept} | {prediction} | {is_correct}")

                time.sleep(3)

            except Exception as e:
                logger.warning(f"Error: {e}")

    if not results:
        logger.warning("No results collected. Check API key, model availability, or image paths.")
        return

    df = pd.DataFrame(results)

    # Aggregate per concept
    summary = df.groupby(["model", "concept"]).agg({
        "is_correct": "mean",
        "latency": "mean"
    }).reset_index()

    summary.rename(columns={"is_correct": "accuracy"}, inplace=True)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df.to_csv("data/processed/model_raw_results.csv", index=False)
    summary.to_csv("data/processed/model_confidence.csv", index=False)

    logger.info("Saved model confidence results")


if __name__ == "__main__":
    run()