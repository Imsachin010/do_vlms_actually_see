"""
src/runners.py
Unified Inference Runner for HF Local FP16, Cloud APIs, and Local Quantized Engines.
Deterministic Settings: temperature=0.0, seed=42, greedy decoding.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from src.parse import parse_integer
from src.schema import format_evaluation_row


def load_prompt_config(prompt_name: str = "IMAGE_CONDITION") -> str:
    """Load canonical prompt string from configs/prompts.txt."""
    cfg_path = Path("configs/prompts.txt")
    if not cfg_path.exists():
        # Fallback to default
        return "How many legs does this animal exactly have in this image? Do not use prior knowledge about animals. Answer with just a single number integer, nothing else."

    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()

    section = f"[{prompt_name}]"
    if section in content:
        lines = []
        capture = False
        for line in content.splitlines():
            if line.strip().startswith("[") and line.strip().endswith("]"):
                if line.strip() == section:
                    capture = True
                    continue
                elif capture:
                    break
            elif capture and line.strip():
                lines.append(line.strip())
        return " ".join(lines)

    return "How many legs does this animal exactly have in this image? Do not use prior knowledge about animals. Answer with just a single number integer, nothing else."


def run_groq_query(
    image_b64: str,
    prompt: str,
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    api_key: Optional[str] = None
) -> str:
    """Execute inference query via Groq API endpoint."""
    key = api_key or os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 20
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"].strip()
    return ""


def run_ollama_query(
    prompt: str,
    image_b64: Optional[str] = None,
    model_name: str = "llava:7b",
    endpoint: str = "http://localhost:11434/api/generate"
) -> str:
    """Execute inference query via local Ollama engine."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42}
    }
    if image_b64:
        payload["images"] = [image_b64]

    r = requests.post(endpoint, json=payload, timeout=60)
    if r.status_code == 200:
        return r.json().get("response", "").strip()
    return ""
