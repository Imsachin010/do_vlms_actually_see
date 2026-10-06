import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import time

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODELS = {
    "Gemini-Flash": "models/gemini-flash-latest",
    "Gemini-2.5-Pro": "models/gemini-2.5-pro"
}

PROMPT = "How many legs does this animal have? Answer with just a number."

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

def run():
    image_dir = "data/raw/counterfactual_images"
    results = []
    
    csv_path = Path("data/processed/gemini_results.csv")
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        results = df_existing.to_dict('records')
        print(f"Resuming from {len(results)} existing records...")
    
    processed_keys = set(f"{r['model']}_{r['image']}" for r in results)

    for model_name, model_id in MODELS.items():
        print(f"Running model: {model_name}")
        model = genai.GenerativeModel(model_id)

        for file in os.listdir(image_dir):
            if not file.endswith((".png", ".jpg")):
                continue
                
            if f"{model_name}_{file}" in processed_keys:
                continue

            concept, true_legs, bg = parse_filename(file)
            path = os.path.join(image_dir, file)
            image_obj = Image.open(path)

            try:
                time.sleep(13.0)
                response = model.generate_content([image_obj, PROMPT])
                response.resolve()
                text = response.text
            except Exception as e:
                print(f"[{model_name}] FAILED on {file}: {e}")
                text = ""

            pred = extract_number(text)
            error = abs(pred - true_legs) if pred is not None else None
            is_biased = 0
            if pred is not None:
                if abs(pred - 4) < abs(true_legs - 4):
                     is_biased = 1

            print(f"{model_name} | {file} | pred={pred} | true={true_legs} | bias={is_biased}")

            results.append({
                "model": model_name,
                "concept": concept,
                "image": file,
                "true_legs": true_legs,
                "predicted": pred,
                "is_biased": is_biased,
                "error": error,
                "background": bg
            })
            
            pd.DataFrame(results).to_csv(csv_path, index=False)
            processed_keys.add(f"{model_name}_{file}")

    print("Saved Phase 2 Gemini results")

if __name__ == "__main__":
    run()
