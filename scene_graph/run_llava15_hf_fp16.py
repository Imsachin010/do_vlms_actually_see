"""
Run LLaVA-1.5 (7B) HuggingFace Local FP16 benchmark on 60 Scene Graph v2 JSONs.

Model ID: llava-hf/llava-1.5-7b-hf
Prompt: Same fixed SG prompt, temperature=0, seed=42
"""

import gc
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

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


def run_llava15_hf_fp16():
    model_id = "llava-hf/llava-1.5-7b-hf"
    slug = "llava_15_hf_fp16"
    print("============================================================")
    print(f"RUNNING LOCAL TRANSFORMERS: LLaVA-1.5-7B FP16 ({model_id})")
    print("============================================================")

    out_csv = RESULTS_DIR / f"{slug}.csv"

    ann_path = DATA_DIR / "annotations_v2.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    dummy_img = Image.new("RGB", (336, 336), color="white")

    records = []
    for idx, ann in enumerate(annotations, 1):
        image_id = ann["image_id"]
        sg_path = DATA_DIR / "scene_graphs_v2" / f"{image_id}.json"
        with open(sg_path, "r", encoding="utf-8") as f:
            sg = json.load(f)

        prompt = PROMPT_TEMPLATE.format(scene_graph_json=json.dumps(sg, indent=2))
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(text=full_prompt, images=dummy_img, return_tensors="pt").to(model.device)

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
            "model": "LLaVA-1.5-7B (HF Local FP16)",
        }
        records.append(record)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"  [{idx:2d}/60] {image_id:30s} | parsed={parsed} | true={true_legs} | correct={correct} | raw={repr(raw_output[:40])}")

    acc = (sum(r["correct"] for r in records) / len(records) * 100)
    print(f"\n[OK] LLaVA-1.5-7B HF Local FP16 Complete: 60/60 Valid | Accuracy = {acc:.1f}% ({sum(r['correct'] for r in records)}/60)")


if __name__ == "__main__":
    run_llava15_hf_fp16()
