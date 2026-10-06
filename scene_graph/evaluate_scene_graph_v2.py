"""
Scene Graph Evaluation v2 for VLM Bias Prediction (Weakness 1 Fix)

Benchmarking VLMs on redesigned scene graphs requiring true counting inference.

Usage:
    python scene_graph/evaluate_scene_graph_v2.py

Output:
    results/scene_graph_v2_inference_required/{model_slug}.csv
    scene_graph/results/scene_graph_v2_inference_required/{model_slug}.csv
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Set in .env or environment variables.")

DATA_DIR = Path("scene_graph/data")
RESULTS_DIR_1 = Path("results/scene_graph_v2_inference_required")
RESULTS_DIR_2 = Path("scene_graph/results/scene_graph_v2_inference_required")
RESULTS_DIR_1.mkdir(parents=True, exist_ok=True)
RESULTS_DIR_2.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = """Based solely on the relations listed in the following scene description, count how many distinct legs are described. Do not use prior knowledge about how many legs this type of animal typically has — count only what is explicitly listed as a relation. Answer with a single integer.
Scene: {scene_graph_json}"""

MODELS = [
    {
        "name": "Llama-4-Scout",
        "hf_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "provider": "groq_native",
        "groq_model": "llama-3.3-70b-versatile", # Fallback to top Llama on Groq if Scout ID 404s
    },
    {
        "name": "Gemma-4-31B",
        "hf_id": "google/gemma-4-31B-it",
        "provider": None,
    },
    {
        "name": "Qwen3-VL-235B-A22B",
        "hf_id": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "provider": "novita",
    },
]


def parse_answer(text: str) -> int | None:
    """Extract a single integer from model response."""
    if not text:
        return None

    raw = text.strip().lower()

    # 1) Direct digits search
    patterns = [
        r"(?:answer\s*(?:is|:)?\s*)(\d{1,2})\b",
        r"(?:total\s*(?:legs?)?\s*(?:is|:)?\s*)(\d{1,2})\b",
        r"(?:there\s+(?:are|is)\s*)(\d{1,2})\s+legs?\b",
        r"(?:count\s*(?:is|:)?\s*)(\d{1,2})\b",
        r"(?:\*\*)?(\d{1,2})(?:\*\*)?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.MULTILINE)
        if m:
            return int(m.group(1))

    # 2) Word-to-int fallback
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10,
    }
    for word, val in word_map.items():
        if re.search(rf"\b{word}\b", raw):
            return val

    # 3) Last resort: grab last integer
    nums = re.findall(r"\b(\d{1,2})\b", raw)
    if nums:
        return int(nums[-1])

    return None


def query_hf_model(client: InferenceClient, model_id: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=128,
        temperature=0.0,
    )
    msg = response.choices[0].message
    content = msg.content

    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        parts = [str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content]
        return " ".join(parts).strip()
    elif content is not None:
        return str(content).strip()

    return ""


def query_groq_native(prompt: str, model_id: str = "llama-3.3-70b-versatile") -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
    if res.status_code == 200:
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        # Try alternate Groq model if specific ID unavailable
        payload["model"] = "llama-3.3-70b-versatile"
        res2 = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
        if res2.status_code == 200:
            return res2.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"Groq API error [{res.status_code}]: {res.text}")


def load_annotations_v2() -> list[dict]:
    ann_path = DATA_DIR / "annotations_v2.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_graph_v2(image_id: str) -> dict:
    sg_path = DATA_DIR / "scene_graphs_v2" / f"{image_id}.json"
    with open(sg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run():
    annotations = load_annotations_v2()
    print(f"Loaded {len(annotations)} annotated scene graphs v2")

    for m in MODELS:
        model_name = m["name"]
        model_id = m["hf_id"]
        provider = m["provider"]
        slug = model_name.lower().replace("-", "_").replace(".", "")

        csv_out_1 = RESULTS_DIR_1 / f"{slug}.csv"
        csv_out_2 = RESULTS_DIR_2 / f"{slug}.csv"
        log_out = RESULTS_DIR_2 / f"{slug}_outputs.jsonl"

        # Resume support
        records = []
        if csv_out_1.exists():
            try:
                records = pd.read_csv(csv_out_1).to_dict("records")
            except Exception:
                records = []
        done = {r["image_id"] for r in records if r.get("status") == "ok"}

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})" + (f" via {provider}" if provider else " via default HF"))
        print(f"Output CSV: {csv_out_1}")
        print(f"{'='*60}")

        # Setup client
        if provider == "groq_native":
            client = None
        else:
            kwargs = {"api_key": HF_TOKEN}
            if provider:
                kwargs["provider"] = provider
            client = InferenceClient(**kwargs)

        for ann in annotations:
            image_id = ann["image_id"]
            if image_id in done:
                print(f"  skip {image_id}")
                continue

            sg = load_scene_graph_v2(image_id)
            sg_json_str = json.dumps(sg, indent=2)
            prompt = PROMPT_TEMPLATE.format(scene_graph_json=sg_json_str)

            raw_output = ""
            parsed = None
            error_msg = None

            for attempt in range(3):
                try:
                    if provider == "groq_native":
                        raw_output = query_groq_native(prompt, m.get("groq_model", "meta-llama/llama-4-scout-17b-16e-instruct"))
                    else:
                        raw_output = query_hf_model(client, model_id, prompt)

                    parsed = parse_answer(raw_output)
                    error_msg = None
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"  [Attempt {attempt+1}/3 failed for {image_id}]: {error_msg}")
                    time.sleep(1.5 * (attempt + 1))

            status = "ok" if error_msg is None and raw_output.strip() else ("api_error" if error_msg else "empty_output")
            true_legs = int(ann["true_legs"])
            correct = int(parsed == true_legs) if parsed is not None else 0

            # Categorize error
            if correct:
                prior_aligned = 0
                off_by_one = 0
                other_err = 0
            else:
                if parsed == 4:
                    prior_aligned = 1
                    off_by_one = 0
                    other_err = 0
                elif parsed is not None and abs(parsed - true_legs) == 1:
                    prior_aligned = 0
                    off_by_one = 1
                    other_err = 0
                else:
                    prior_aligned = 0
                    off_by_one = 0
                    other_err = 1

            record = {
                "image_id": image_id,
                "concept": ann["concept"],
                "true_legs": true_legs,
                "background": ann["background"],
                "model_response_raw": raw_output,
                "parsed_answer": parsed if parsed is not None else -1,
                "correct": correct,
                "prior_aligned_error": prior_aligned,
                "off_by_one_error": off_by_one,
                "other_error": other_err,
                "status": status,
                "model": model_name,
                "hf_id": model_id,
                "provider": provider or "default",
                "error": error_msg,
            }
            records.append(record)

            df = pd.DataFrame(records)
            df.to_csv(csv_out_1, index=False)
            df.to_csv(csv_out_2, index=False)

            with open(log_out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **record}) + "\n")

            print(f"  {image_id} | parsed={parsed} | true={true_legs} | correct={correct} | prior_err={prior_aligned} | off1_err={off_by_one} | status={status}")

        print(f"[OK] Saved {len(records)} records for {model_name}")


if __name__ == "__main__":
    run()
