"""
Llama 4 Scout — Local runner using Groq API
Images: data/raw/counterfactual_images/
Output: data/processed/llama4_scout_local_results.csv
"""

import os, re, base64, time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parents[1]
IMAGE_DIR = BASE / "data" / "raw" / "counterfactual_images"
OUT_CSV   = BASE / "data" / "processed" / "llama4_scout_local_results.csv"

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
PROMPT   = """Look at this image carefully.

How many legs does this animal have?
Count ALL visible legs.

Respond with only a number."""

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────
def get_images():
    return sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

def parse_filename(fname):
    name = Path(fname).stem
    m = re.match(r'^(.+?)_(\d+)legs_(natural|plain)$', name)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return name, None, None

def extract_number(text):
    if not text: return None
    m = re.search(r'\d+', text.lower())
    if m: return int(m.group())
    for w, n in {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6}.items():
        if w in text.lower(): return n
    return None

def groq_infer(image_path, retries=3):
    ext  = str(image_path).split(".")[-1].lower()
    mime = "image/jpeg" if ext in ("jpg","jpeg") else "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text",      "text": PROMPT}
                ]}],
                max_tokens=10,
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [attempt {attempt+1}] {e}")
            time.sleep(5 * (attempt + 1))
    return None

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    records = list(pd.read_csv(OUT_CSV).to_dict('records')) if OUT_CSV.exists() else []
    done    = {r['image'] for r in records}
    images  = get_images()

    print(f"Found {len(images)} images | {len(done)} already done")

    for fname in images:
        if fname in done:
            print(f"  skip {fname}"); continue

        concept, true_legs, bg = parse_filename(fname)
        response = groq_infer(IMAGE_DIR / fname)

        if response is None:
            print(f"  ❌ FAILED: {fname}"); continue

        pred = extract_number(response)
        is_biased = int(pred != true_legs) if pred is not None else 1

        records.append({"image": fname, "concept": concept, "true_legs": true_legs,
                        "background": bg, "predicted": pred, "is_biased": is_biased,
                        "model": "Llama4-Scout", "hf_id": MODEL_ID})
        pd.DataFrame(records).to_csv(OUT_CSV, index=False)
        print(f"Llama4-Scout | {fname} | raw='{response}' | pred={pred} | true={true_legs} | bias={is_biased}")

        time.sleep(2.5)

    print(f"\n✅ Done. Results saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()
