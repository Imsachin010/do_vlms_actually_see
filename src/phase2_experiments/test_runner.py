import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import time
from pathlib import Path
from src.utils.logger import setup_logger
from src.phase1_prior_strength.groq_client import GroqVLM

logger = setup_logger()

API_KEY = os.getenv("GROQ_API_KEY")

MODELS = {
    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    #"llava": "llava-v1.5-7b-4096-preview",
    #"qwen": "qwen-2.5-vl-7b-instruct",
    
}


PROMPT = """
Look at this image carefully.

How many legs does this animal have?
Count ALL visible legs.

Respond with only a number.
"""


def extract_number(text):
    import re
    text = text.lower()

    match = re.search(r"\d+", text)
    if match:
        return int(match.group())

    word_map = {
        "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6
    }

    for w, n in word_map.items():
        if w in text:
            return n

    return None


def parse_filename(filename):
    parts = filename.replace(".png", "").replace(".jpg", "").split("_")
    concept = parts[0]
    legs = int(parts[1].replace("legs", ""))
    background = parts[2]
    return concept, legs, background


def run():
    image_dir = "data/raw/counterfactual_images"
    results = []

    for model_name, model_id in MODELS.items():
        logger.info(f"Running model: {model_name}")
        client = GroqVLM(API_KEY, model_id)

        for file in os.listdir(image_dir):
            if not file.endswith((".png", ".jpg")):
                continue

            concept, true_legs, bg = parse_filename(file)
            path = os.path.join(image_dir, file)

            try:
                response, latency = client.query(path, PROMPT)

                if response is None:
                    continue

                pred = extract_number(response)

                is_correct = int(pred == true_legs)
                error = abs(pred - true_legs) if pred is not None else None

                is_biased = 0
                if pred is not None:
                    if abs(pred - 4) < abs(true_legs - 4):
                        is_biased = 1

                results.append({
                    "model": model_name,
                    "concept": concept,
                    "image": file,
                    "true_legs": true_legs,
                    "predicted": pred,
                    "is_correct": is_correct,
                    "is_biased": is_biased,
                    "error": error,
                    "background": bg,
                    "latency": latency,
                })

                logger.info(f"{model_name} | {file} | pred={pred} | true={true_legs}")

                time.sleep(2.5)

            except Exception as e:
                logger.warning(f"Error: {e}")

    df = pd.DataFrame(results)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/phase2_vlm_results.csv", index=False)

    logger.info("Saved Phase 2 VLM results")


if __name__ == "__main__":
    run()