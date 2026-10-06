"""
Scene Graph v2 Benchmark Runner for 10 Models

Evaluates 10 VLMs on 60 leakage-verified Scene Graph JSONs (enumeration-required design, no digits/ordinals).

Models:
  1. LLaVA-1.5 (HF)         -> llava-hf/llava-1.5-7b-hf (HF)
  2. LLaVA-1.5 (Ollama)     -> llava:7b (Ollama Q4_0)
  3. LLaVA-v1.6-Mistral     -> llava-hf/llava-v1.6-mistral-7b-hf (HF)
  4. InstructBLIP-T5XL      -> Salesforce/instructblip-flant5-xl (HF)
  5. Llama-3.2-Vision       -> llama3.2-vision (Ollama Q4_0)
  6. MiniCPM-V             -> minicpm-v (Ollama Q4_0)
  7. SmolVLM2              -> HuggingFaceTB/SmolVLM2-2.2B-Instruct (HF)
  8. Qwen2-VL              -> Qwen/Qwen2-VL-2B-Instruct (HF)
  9. Qwen2.5-VL            -> Qwen/Qwen2.5-VL-7B-Instruct (HF)
 10. Gemma-3-4B            -> google/gemma-3-4b-it (HF)
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
OLLAMA_URL = "http://localhost:11434"

DATA_DIR = Path("scene_graph/data")
RESULTS_DIR_1 = Path("results/scene_graph_v2_inference_required")
RESULTS_DIR_2 = Path("scene_graph/results/scene_graph_v2_inference_required")
RESULTS_DIR_1.mkdir(parents=True, exist_ok=True)
RESULTS_DIR_2.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = """Based solely on the relations listed in the following scene description, count how many distinct legs are described. Do not use prior knowledge about how many legs this type of animal typically has — count only what is explicitly listed as a relation. Answer with a single integer.
Scene: {scene_graph_json}"""

MODELS_CONFIG = [
    {
        "name": "LLaVA-1.5 (HF)",
        "slug": "llava_15_hf",
        "engine": "hf",
        "model_id": "llava-hf/llava-1.5-7b-hf",
    },
    {
        "name": "LLaVA-1.5 (Ollama)",
        "slug": "llava_15_ollama",
        "engine": "ollama",
        "model_id": "llava:7b",
    },
    {
        "name": "LLaVA-v1.6-Mistral",
        "slug": "llava_v16_mistral",
        "engine": "hf",
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
    },
    {
        "name": "InstructBLIP-T5XL",
        "slug": "instructblip_t5xl",
        "engine": "hf",
        "model_id": "Salesforce/instructblip-flant5-xl",
    },
    {
        "name": "Llama-3.2-Vision",
        "slug": "llama_32_vision",
        "engine": "ollama",
        "model_id": "llama3.2-vision",
    },
    {
        "name": "MiniCPM-V",
        "slug": "minicpm_v",
        "engine": "ollama",
        "model_id": "minicpm-v",
    },
    {
        "name": "SmolVLM2",
        "slug": "smolvlm2",
        "engine": "hf",
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    },
    {
        "name": "Qwen2-VL",
        "slug": "qwen2_vl",
        "engine": "hf",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
    },
    {
        "name": "Qwen2.5-VL",
        "slug": "qwen25_vl",
        "engine": "hf",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    {
        "name": "Gemma-3-4B",
        "slug": "gemma_3_4b",
        "engine": "hf",
        "model_id": "google/gemma-3-4b-it",
    },
]


def parse_answer(text: str) -> int | None:
    if not text:
        return None
    raw = text.strip().lower()

    # Digits search
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

    # Word fallback
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10,
    }
    for word, val in word_map.items():
        if re.search(rf"\b{word}\b", raw):
            return val

    nums = re.findall(r"\b(\d{1,2})\b", raw)
    if nums:
        return int(nums[-1])

    return None


def query_ollama(model_id: str, prompt: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42
        }
    }
    res = requests.post(url, json=payload, timeout=60)
    if res.status_code == 200:
        return res.json().get("response", "").strip()
    else:
        raise RuntimeError(f"Ollama HTTP {res.status_code}: {res.text}")


def query_hf(client: InferenceClient, model_id: str, prompt: str) -> str:
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


def ensure_ollama_model(model_id: str):
    import subprocess
    res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    installed = [m.get("name") for m in res.json().get("models", [])]
    if model_id in installed or f"{model_id}:latest" in installed:
        print(f"[Ollama] Model {model_id} already installed.")
        return

    print(f"[Ollama] Pulling model {model_id} via CLI subprocess...")
    proc = subprocess.run(["ollama", "pull", model_id], capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"[Ollama] Pull {model_id} complete!")
    else:
        print(f"[WARN] Failed to pull {model_id}: {proc.stderr}")


def load_annotations_v2() -> list[dict]:
    ann_path = DATA_DIR / "annotations_v2.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_graph_v2(image_id: str) -> dict:
    sg_path = DATA_DIR / "scene_graphs_v2" / f"{image_id}.json"
    with open(sg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_model(m_cfg: dict, annotations: list[dict], hf_client: InferenceClient | None):
    name = m_cfg["name"]
    slug = m_cfg["slug"]
    engine = m_cfg["engine"]
    model_id = m_cfg["model_id"]

    csv_out_1 = RESULTS_DIR_1 / f"{slug}.csv"
    csv_out_2 = RESULTS_DIR_2 / f"{slug}.csv"
    log_out = RESULTS_DIR_2 / f"{slug}_outputs.jsonl"

    print(f"\n============================================================")
    print(f"STARTING EVALUATION: {name} ({model_id}) [{engine.upper()}]")
    print(f"============================================================")

    if engine == "ollama":
        ensure_ollama_model(model_id)

    records = []
    if csv_out_1.exists():
        try:
            records = pd.read_csv(csv_out_1).to_dict("records")
        except Exception:
            records = []

    done_ids = {r["image_id"] for r in records if r.get("status") == "ok"}
    print(f"Progress: {len(done_ids)}/60 completed for {name}")

    for ann in annotations:
        image_id = ann["image_id"]
        if image_id in done_ids:
            continue

        sg = load_scene_graph_v2(image_id)
        sg_json_str = json.dumps(sg, indent=2)
        prompt = PROMPT_TEMPLATE.format(scene_graph_json=sg_json_str)

        raw_output = ""
        parsed = None
        error_msg = None

        for attempt in range(3):
            try:
                if engine == "ollama":
                    raw_output = query_ollama(model_id, prompt)
                else:
                    raw_output = query_hf(hf_client, model_id, prompt)

                parsed = parse_answer(raw_output)
                error_msg = None
                break
            except Exception as e:
                error_msg = str(e)
                print(f"  [Attempt {attempt+1}/3 failed for {image_id}]: {error_msg}")
                time.sleep(2.0 * (attempt + 1))

        status = "ok" if error_msg is None and raw_output.strip() else ("api_error" if error_msg else "empty_output")
        true_legs = int(ann["true_legs"])
        correct = int(parsed == true_legs) if parsed is not None else 0

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
            "off-by-one_error": off_by_one,
            "other_error": other_err,
            "status": status,
            "model": name,
            "model_id": model_id,
            "engine": engine,
            "error": error_msg,
        }
        records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(csv_out_1, index=False)
        df.to_csv(csv_out_2, index=False)

        with open(log_out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **record}) + "\n")

        print(f"  [{len(records)}/60] {image_id:30s} | parsed={parsed} | true={true_legs} | correct={correct} | status={status}")

    acc = sum(r["correct"] for r in records) / len(records) if records else 0.0
    print(f"\n[OK] COMPLETED {name}: Accuracy = {acc:.1%} ({sum(r['correct'] for r in records)}/{len(records)})")


def main():
    target_slug = sys.argv[1] if len(sys.argv) > 1 else None

    annotations = load_annotations_v2()
    hf_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

    for m_cfg in MODELS_CONFIG:
        if target_slug and m_cfg["slug"] != target_slug:
            continue
        run_single_model(m_cfg, annotations, hf_client)

if __name__ == "__main__":
    main()
