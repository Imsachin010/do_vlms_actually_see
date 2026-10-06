import base64
import requests
import json
import pandas as pd
import os
from pathlib import Path

MODEL_ID = "llava:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = "How many legs does this animal have? Answer with a number."

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
    return parts[0], int(parts[1].replace("legs", "")), parts[2]

def run():
    image_dir = "data/raw/counterfactual_images"
    results = []

    for file in os.listdir(image_dir):
        if not file.endswith((".png", ".jpg")):
            continue

        concept, true_legs, bg = parse_filename(file)
        path = os.path.join(image_dir, file)
        
        image_b64 = encode_image(path)

        payload = {
            "model": MODEL_ID,
            "prompt": PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 20}
        }
        
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response = resp.json().get("response", "")
        except Exception as e:
            print(f"Failed to query Ollama for {file}: {e}")
            continue

        pred = extract_number(response)

        error = abs(pred - true_legs) if pred is not None else None
        is_biased = 0
        if pred is not None:
            if abs(pred - 4) < abs(true_legs - 4):
                 is_biased = 1

        print(f"LLaVA 1.5 | {file} | pred={pred} | true={true_legs} | bias={is_biased}")

        results.append({
            "model": "llava-1.5-7b-ollama",
            "concept": concept,
            "image": file,
            "true_legs": true_legs,
            "predicted": pred,
            "is_biased": is_biased,
            "error": error
        })

    df = pd.DataFrame(results)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/llava_results.csv", index=False)
    print("Saved Phase 2 LLaVA results in data/processed/llava_results.csv")

if __name__ == "__main__":
    run()
