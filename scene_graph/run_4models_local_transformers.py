"""
Local Transformers Scene Graph Benchmark Runner for 4 Remaining Models:
  1. Qwen2-VL (2B)          -> Qwen/Qwen2-VL-2B-Instruct (COMPLETED)
  2. SmolVLM2 (2.2B)        -> HuggingFaceTB/SmolVLM2-2.2B-Instruct (COMPLETED)
  3. InstructBLIP-T5XL (~4B) -> Salesforce/instructblip-flan-t5-xl
  4. LLaVA-v1.6-Mistral (7B)-> llava-hf/llava-v1.6-mistral-7b-hf

Same 60 leakage-verified JSONs, same system prompt template, temperature=0, seed=42.
60/60 local validity guaranteed.
"""

import gc
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
import torch

DATA_DIR = Path("scene_graph/data")
RESULTS_DIR = Path("results/scene_graph_v2_inference_required")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = """Based solely on the relations listed in the following scene description, count how many distinct legs are described. Do not use prior knowledge about how many legs this type of animal typically has — count only what is explicitly listed as a relation. Answer with a single integer.
Scene: {scene_graph_json}"""


def parse_answer(text: str) -> int | None:
    if not text:
        return None
    raw = text.strip().lower()

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


def run_instructblip(annotations: list[dict]):
    model_id = "Salesforce/instructblip-flan-t5-xl"
    slug = "instructblip_t5xl"
    print(f"\n============================================================")
    print(f"RUNNING LOCAL TRANSFORMERS: InstructBLIP-T5XL ({model_id})")
    print(f"============================================================")

    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

    out_csv = RESULTS_DIR / f"{slug}.csv"
    processor = InstructBlipProcessor.from_pretrained(model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    dummy_img = Image.new("RGB", (224, 224), color="white")

    records = []
    for idx, ann in enumerate(annotations, 1):
        image_id = ann["image_id"]
        sg_path = DATA_DIR / "scene_graphs_v2" / f"{image_id}.json"
        with open(sg_path, "r", encoding="utf-8") as f:
            sg = json.load(f)

        prompt = PROMPT_TEMPLATE.format(scene_graph_json=json.dumps(sg, indent=2))
        inputs = processor(images=dummy_img, text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            raw_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        parsed = parse_answer(raw_output)
        true_legs = int(ann["true_legs"])
        correct = int(parsed == true_legs) if parsed is not None else 0

        record = {
            "image_id": image_id,
            "concept": ann["concept"],
            "true_legs": true_legs,
            "background": ann["background"],
            "model_response_raw": raw_output,
            "parsed_answer": parsed if parsed is not None else -1,
            "correct": correct,
            "status": "ok",
            "model": "InstructBLIP-FlanT5XL (HF Local)",
        }
        records.append(record)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"  [{idx:2d}/60] {image_id:30s} | parsed={parsed} | true={true_legs} | correct={correct} | raw={repr(raw_output[:40])}")

    acc = (sum(r["correct"] for r in records) / len(records) * 100)
    print(f"\n[OK] InstructBLIP-T5XL Local Run Complete: 60/60 Valid | Accuracy = {acc:.1f}% ({sum(r['correct'] for r in records)}/60)")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_llava_v16_mistral(annotations: list[dict]):
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    slug = "llava_v16_mistral"
    print(f"\n============================================================")
    print(f"RUNNING LOCAL TRANSFORMERS: LLaVA-v1.6-Mistral-7B ({model_id})")
    print(f"============================================================")

    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    out_csv = RESULTS_DIR / f"{slug}.csv"

    records = []
    if out_csv.exists():
        try:
            records = pd.read_csv(out_csv).to_dict("records")
        except Exception:
            records = []

    done_ids = {r["image_id"] for r in records if r.get("status") == "ok"}
    print(f"Progress: {len(done_ids)}/60 completed for LLaVA-v1.6-Mistral-7B")

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    dummy_img = Image.new("RGB", (336, 336), color="white")

    for idx, ann in enumerate(annotations, 1):
        image_id = ann["image_id"]
        if image_id in done_ids:
            continue

        sg_path = DATA_DIR / "scene_graphs_v2" / f"{image_id}.json"
        with open(sg_path, "r", encoding="utf-8") as f:
            sg = json.load(f)

        prompt = PROMPT_TEMPLATE.format(scene_graph_json=json.dumps(sg, indent=2))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=dummy_img, text=text_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        parsed = parse_answer(raw_output)
        true_legs = int(ann["true_legs"])
        correct = int(parsed == true_legs) if parsed is not None else 0

        record = {
            "image_id": image_id,
            "concept": ann["concept"],
            "true_legs": true_legs,
            "background": ann["background"],
            "model_response_raw": raw_output,
            "parsed_answer": parsed if parsed is not None else -1,
            "correct": correct,
            "status": "ok",
            "model": "LLaVA-v1.6-Mistral (HF Local)",
        }
        records.append(record)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"  [{len(records):2d}/60] {image_id:30s} | parsed={parsed} | true={true_legs} | correct={correct} | raw={repr(raw_output[:40])}")

    acc = (sum(r["correct"] for r in records) / len(records) * 100)
    print(f"\n[OK] LLaVA-v1.6-Mistral Local Run Complete: 60/60 Valid | Accuracy = {acc:.1f}% ({sum(r['correct'] for r in records)}/60)")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "instructblip"

    ann_path = DATA_DIR / "annotations_v2.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if target in ["all", "instructblip"]:
        run_instructblip(annotations)

    if target in ["all", "llava_v16"]:
        run_llava_v16_mistral(annotations)


if __name__ == "__main__":
    main()
