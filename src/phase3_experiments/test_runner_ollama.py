import os
import time
import requests
import base64
import pandas as pd
from pathlib import Path

MODELS = ["llava:7b", "minicpm-v", "llama3.2-vision"]
OLLAMA_API = "http://localhost:11434/api/generate"

PROMPT = "How many legs does this animal exactly have in this image? Answer with just a single number integer, nothing else."

def extract_number(text):
    import re
    if not text:
        return None
    text = text.lower()
    
    match = re.search(r"\d+", text)
    if match: return int(match.group())
    
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    for w, n in word_map.items():
        if w in text: return n
    return None

def parse_filename(filename):
    parts = filename.replace(".png", "").replace(".jpg", "").split("_")
    return parts[0], int(parts[1].replace("legs", "")), parts[2]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run():
    image_dir = "data/raw/counterfactual_images"
    os.makedirs("data/processed", exist_ok=True)
    csv_path = Path("data/processed/phase3_ollama_results.csv")
    
    results = []
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        results = df_existing.to_dict('records')
        print(f"Resuming Phase 3 from {len(results)} existing base records...")
        
    processed_keys = set(f"{r['model']}_{r['image']}" for r in results)

    # Serial inference to prevent multi-model VRAM explosions
    for model in MODELS:
        print(f"\n--- Evaluating {model} ---")
        
        for file in os.listdir(image_dir):
            if not file.endswith((".png", ".jpg")): continue
            
            key = f"{model}_{file}"
            if key in processed_keys:
                continue

            concept, true_legs, bg = parse_filename(file)
            path = os.path.join(image_dir, file)
            base64_str = encode_image(path)

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT,
                        "images": [base64_str]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "seed": 42
                }
            }

            try:
                # 5 minute timeout since quant models may take 30s to load into VRAM
                response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=300)
                if response.status_code == 200:
                    text = response.json().get("message", {}).get("content", "")
                else:
                    print(f"[{model}] API Error {response.status_code}: {response.text}")
                    text = ""
            except Exception as e:
                print(f"[{model}] Request failed tightly on {file}: {e}")
                text = ""

            pred = extract_number(text)
            error = abs(pred - true_legs) if pred is not None else None
            
            is_biased = 0
            if pred is not None:
                # User's strict definition: anything other than true_legs is false/bias
                if pred != true_legs:
                     is_biased = 1

            print(f"{model} | {file} | pred={pred} | true={true_legs} | bias={is_biased}")

            results.append({
                "model": model,
                "concept": concept,
                "image": file,
                "true_legs": true_legs,
                "predicted": pred,
                "is_biased": is_biased,
                "error": error,
                "background": bg
            })
            
            # Atomic save state guarantees failure resilience
            pd.DataFrame(results).to_csv(csv_path, index=False)
            processed_keys.add(key)
            
    print("\nPhase 3 Core Evaluation Complete!")

if __name__ == "__main__":
    run()
