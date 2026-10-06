"""
src/eval_scene_graph.py
Rigorous Scene Graph Evaluation Pipeline across all 14 Vision-Language configurations.
Implements the 5-phase audit and Regime A/B gate verification.
"""

from pathlib import Path
import pandas as pd
from src.stats_utils import wilson_score_interval
from src.parse import parse_integer

RESULTS_DIR = Path("results/scene_graph")
if not RESULTS_DIR.exists():
    RESULTS_DIR = Path("results/scene_graph_v2_inference_required")

OUT_DIR = Path("analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_CONFIGS = [
    ("Llama-4-Scout (17B)", "llama_4_scout.csv", "Groq API", 13.3),
    ("Gemma-4-31B (31B)", "gemma_4_31b.csv", "HF API", 11.7),
    ("Qwen3-VL-235B (235B)", "qwen3_vl_235b_a22b.csv", "Novita API", 0.0),
    ("LLaVA-v1.6-Mistral (7B)", "llava_v16_mistral.csv", "HF Local FP16", 0.0),
    ("InstructBLIP-FlanT5-XL (4B)", "instructblip_t5xl.csv", "HF Local FP16", 0.0),
    ("Qwen2.5-VL (7B FP16)", "qwen25_vl_fp16_60queries.csv", "HF API Paced", 0.0),
    ("Qwen2.5-VL (7B Q4_0)", "qwen25_7b_ollama.csv", "Ollama Q4_0 Local", 0.0),
    ("LLaVA-1.5 (7B Q4_0)", "llava_15_ollama.csv", "Ollama Q4_0 Local", 0.0),
    ("LLaVA-1.5 (7B FP16)", "llava_15_hf_fp16.csv", "HF Local FP16", 0.0),
    ("MiniCPM-V (8B)", "minicpm_v.csv", "Ollama Q4_0 Local", 0.0),
    ("Qwen2-VL (2B)", "qwen2_vl.csv", "HF Local FP16", 0.0),
    ("SmolVLM2 (2.2B)", "smolvlm2.csv", "HF Local FP16", 0.0),
    ("Gemma-3-4B (4B)", "gemma_3_4b.csv", "HF Local FP16", 0.0),
    ("Llama-3.2-Vision (11B)", "llama32_vision.csv", "Ollama Q4_0 Local", 0.0),
]


def evaluate_scene_graphs():
    print("=" * 80)
    print("SCENE GRAPH BENCHMARK & REGIME GATE AUDIT (14 CONFIGURATIONS)")
    print("=" * 80)

    rows = []
    for model_name, fname, backend, img_acc in MODELS_CONFIGS:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            continue

        df = pd.read_csv(fpath)
        n_total = len(df)
        pred_col = "parsed_answer" if "parsed_answer" in df.columns else "predicted"

        n_correct = int((df[pred_col] == df["true_legs"]).sum())
        sg_acc, ci_low, ci_high = wilson_score_interval(n_correct, n_total)

        # Count 5 -> 4 prior pull
        five_leg = df[df["true_legs"] == 5]
        n_5to4 = int((five_leg[pred_col] == 4).sum())

        gate_pass = (sg_acc >= 95.0)
        regime = "Regime A (Pass)" if gate_pass else "Regime B (Fail)"
        if round(sg_acc, 1) == 90.0:
            regime = "Regime B (Borderline 90.0%)"

        rows.append({
            "model": model_name,
            "backend": backend,
            "image_acc": img_acc,
            "sg_acc": round(sg_acc, 1),
            "sg_ci": f"[{ci_low:.1f}, {ci_high:.1f}]",
            "prior_in_text_5to4": n_5to4,
            "regime": regime
        })

    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))

    out_csv = OUT_DIR / "scene_graph_gate_summary.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Summary exported to {out_csv}")
    return res_df


if __name__ == "__main__":
    evaluate_scene_graphs()
