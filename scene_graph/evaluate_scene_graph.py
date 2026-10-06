"""
Scene Graph Evaluation for VLM Bias Prediction

Tests whether VLMs exhibit LESS bias when given structured scene graph input
(vs raw image perception from the main experiment).

Measures:
  - Pure reasoning error: model gets correct scene graph but answers wrong
  - Bias: model overrides scene graph with semantic prior (answers "4")
  - Grounding: model correctly follows the structured description

Usage:
  python scene_graph/evaluate_scene_graph.py

Requires HF_TOKEN in environment or .env file.
"""

import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Set in .env or environment variables.")

DATA_DIR = Path("scene_graph/data")
RESULTS_DIR = Path("scene_graph/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# System prompt: forces model to reason strictly from the scene graph
SYSTEM_PROMPT = """You are given a structured description of a scene.
Answer the question strictly based on this information.
Do NOT use any prior knowledge about animals.
Only use the facts provided in the Scene Graph."""

USER_PROMPT_TEMPLATE = """Scene Graph:
{scene_graph_text}

Question: {question}
Answer strictly based only on the Scene Graph above."""

QUESTION = "How many legs does this animal have?"

# ── Models to test ─────────────────────────────────────────────────────
# These are the VLMs from the main experiment, now tested on text-only scene graphs.
# Models that require multimodal input are replaced with text-only equivalents.

MODELS = [
    # Gemma-4-31B — works with default (no explicit provider) for text-only
    {"name": "Gemma-4-31B",       "hf_id": "google/gemma-4-31B-it"},

    # Qwen3-VL via novita (text-only works)
    {"name": "Qwen3-VL-235B-A22B","hf_id": "Qwen/Qwen3-VL-235B-A22B-Instruct",          "provider": "novita"},

    # Llama-4-Scout via groq (text-only works)
    {"name": "Llama-4-Scout",     "hf_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "provider": "groq"},
]

# ── Text-Only Equivalents ──────────────────────────────────────────────
# Qwen3.5-9B and Kimi-K2.6 are VLMs that require multimodal input and return
# empty responses on text-only chat. We substitute them with text-only LLMs
# from the same capability tier:
#   Qwen3.5-9B  → Qwen3-8B   (same family, text-only variant)
#   Kimi-K2.6   → Llama-3.1-70B (similar reasoning tier)

TEXT_MODELS = [
    {"name": "Qwen3.5-9B-Text",   "hf_id": "Qwen/Qwen3-8B"},
    {"name": "Kimi-K2.6-Text",    "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
]


# ── Helpers ────────────────────────────────────────────────────────────

def format_scene_graph_text(sg: dict) -> str:
    """Convert a scene graph dict into a readable text description."""
    concept = sg["concept"]
    attrs = sg["attributes"].get(concept, {})
    rels = sg["relations"]

    lines = [f"Objects: {', '.join(sg['objects'])}"]
    if attrs:
        attr_str = "; ".join(f"{k}: {v}" for k, v in attrs.items())
        lines.append(f"Attributes ({concept}): {attr_str}")
    if rels:
        lines.append("Relations:")
        for r in rels:
            lines.append(f"  - {r}")

    return "\n".join(lines)


def parse_answer(text: str):
    """Extract a single number from model response."""
    if not text:
        return None

    raw = text.strip().lower()

    # 1) Look for explicit final-count patterns
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

    # 3) Last resort: grab last number
    nums = re.findall(r"\b(\d{1,2})\b", raw)
    if nums:
        return int(nums[-1])

    return None


def load_annotations() -> list[dict]:
    ann_path = DATA_DIR / "annotations.json"
    with open(ann_path) as f:
        return json.load(f)


def load_scene_graph(image_id: str) -> dict:
    sg_path = DATA_DIR / "scene_graphs" / f"{image_id}.json"
    with open(sg_path) as f:
        return json.load(f)


def save_results(records: list[dict], csv_path: Path):
    pd.DataFrame(records).to_csv(csv_path, index=False)


def append_jsonl_log(log_path: Path, payload: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


# ── Inference ──────────────────────────────────────────────────────────

def query_model(client: InferenceClient, model_id: str, messages: list) -> str:
    """Send a chat request and extract text response."""
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=128,
        temperature=0.0,
    )

    msg = response.choices[0].message
    content = msg.content

    # Handle various response formats
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif item:
                parts.append(str(item))
        return " ".join(parts).strip()
    elif content is not None:
        return str(content).strip()

    # Fallback: check reasoning field (Together AI pattern)
    reasoning = getattr(msg, "reasoning", None)
    if reasoning:
        return str(reasoning).strip()

    return ""


# ── Error Classification ──────────────────────────────────────────────
# We analyze WHY the model got it wrong:

def classify_error(predicted: int, true_legs: int) -> str:
    """
    Classify the type of error:
      - "none": correct answer
      - "semantic_bias": model says 4 (the canonical prior for legs)
      - "reasoning_error": model gives wrong number but NOT the prior (4)
      - "other": any other wrong answer
    """
    if predicted is None:
        return "parse_failed"
    if predicted == true_legs:
        return "none"
    if predicted == 4:
        return "semantic_bias"   # collapsed to canonical prior
    return "reasoning_error"     # wrong but not prior-driven


# ── Main Evaluation Loop ───────────────────────────────────────────────

def run():
    annotations = load_annotations()
    print(f"Loaded {len(annotations)} annotated scene graphs")

    for m in MODELS + TEXT_MODELS:
        model_name = m["name"]
        model_id = m["hf_id"]
        provider = m.get("provider")  # None = use default provider
        slug = model_name.lower().replace("-", "_").replace(".", "")

        csv_out = RESULTS_DIR / f"{slug}_sg_results.csv"
        log_out = RESULTS_DIR / f"{slug}_sg_outputs.jsonl"

        # Resume: skip already-processed entries
        records = []
        if csv_out.exists():
            records = pd.read_csv(csv_out).to_dict("records")
        done = {r["image_id"] for r in records}

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})" + (f" via {provider}" if provider else " via default"))
        print(f"Results: {csv_out}")
        print(f"{'='*60}")

        # Create client: with or without explicit provider
        kwargs = {"api_key": HF_TOKEN}
        if provider:
            kwargs["provider"] = provider
        client = InferenceClient(**kwargs)

        for ann in annotations:
            image_id = ann["image_id"]
            if image_id in done:
                print(f"  skip {image_id}")
                continue

            sg = load_scene_graph(image_id)
            sg_text = format_scene_graph_text(sg)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        scene_graph_text=sg_text,
                        question=ann["question"],
                    ),
                },
            ]

            error_msg = None
            try:
                raw_output = query_model(client, model_id, messages)
                predicted = parse_answer(raw_output)
            except Exception as e:
                raw_output = ""
                predicted = None
                error_msg = str(e)

            # Determine status
            if error_msg is not None:
                status = "api_error"
            elif not raw_output.strip():
                status = "empty_output"
            elif predicted is None:
                status = "parse_failed"
            else:
                status = "ok"

            ground_truth = int(ann["ground_truth"])
            is_biased = int(predicted != ground_truth) if predicted is not None else 1
            error_type = classify_error(predicted, ground_truth)

            record = {
                "image_id": image_id,
                "concept": ann["concept"],
                "true_legs": ground_truth,
                "background": ann["background"],
                "predicted": predicted,
                "is_biased": is_biased,
                "error_type": error_type,
                "raw_output": raw_output,
                "status": status,
                "model": model_name,
                "hf_id": model_id,
                "provider": provider or "default",
                "error": error_msg,
                "scene_graph_text": sg_text,
            }
            records.append(record)

            append_jsonl_log(log_out, {
                "ts": datetime.now(timezone.utc).isoformat(),
                **record,
                "messages": messages,
            })

            save_results(records, csv_out)

            if status != "ok":
                print(f"  {image_id} | status={status} | pred={predicted} | true={ground_truth} | type={error_type}")
            else:
                print(f"  {image_id} | pred={predicted} | true={ground_truth} | biased={is_biased} | type={error_type}")

        print(f"Done: {len(records)} records saved to {csv_out}")


if __name__ == "__main__":
    run()
