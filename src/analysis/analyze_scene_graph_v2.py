"""
Analyze Scene Graph v2 (Inference Required) Results & Compute Perception Error Rates

Calculates updated statistics specified in r1.md:
- New SG Accuracy per model
- New Perception Error Rate = 1 - (Image Accuracy / New SG Accuracy)
- Error type breakdown in the SG condition
"""

import json
from pathlib import Path
import pandas as pd

RESULTS_DIR_V2 = Path("results/scene_graph_v2_inference_required")
RESULTS_DIR_V1 = Path("scene_graph/results")
DATA_DIR = Path("data/processed")

MODELS = [
    {"name": "Llama-4-Scout",      "slug": "llama_4_scout",      "image_acc": 0.1333},
    {"name": "Gemma-4-31B",        "slug": "gemma_4_31b",        "image_acc": 0.1190},
    {"name": "Qwen3-VL-235B-A22B", "slug": "qwen3_vl_235b_a22b", "image_acc": 0.0000},
]


def run():
    summary_rows = []

    print("\n" + "="*80)
    print("SCENE GRAPH V2 (INFERENCE REQUIRED) ABLATION ANALYSIS")
    print("="*80)

    for m in MODELS:
        model_name = m["name"]
        slug = m["slug"]
        image_acc = m["image_acc"]

        csv_path = RESULTS_DIR_V2 / f"{slug}.csv"
        if not csv_path.exists():
            print(f"⚠️ Results CSV not found yet: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        total = len(df)
        if total == 0:
            print(f"⚠️ CSV empty for {model_name}")
            continue

        correct_count = df["correct"].sum()
        sg_acc = correct_count / total

        prior_err_count = df["prior_aligned_error"].sum()
        off1_err_count = df["off_by_one_error"].sum()
        other_err_count = df["other_error"].sum()

        prior_err_pct = prior_err_count / total
        off1_err_pct = off1_err_count / total
        other_err_pct = other_err_count / total

        # Perception Error Rate formula from Table tab:ablation
        # Perception Error Rate = 1 - (Image Acc / SG Acc)
        if sg_acc > 0:
            perception_error_rate = 1.0 - (image_acc / sg_acc)
        else:
            perception_error_rate = 1.0

        summary_rows.append({
            "Model": model_name,
            "Total Queries": total,
            "Image Acc": f"{image_acc:.1%}",
            "New SG Acc (v2)": f"{sg_acc:.1%}",
            "Perception Error Rate": f"{perception_error_rate:.1%}",
            "SG Prior-Aligned Errors": f"{prior_err_pct:.1%}",
            "SG Off-by-One Errors": f"{off1_err_pct:.1%}",
            "SG Other Errors": f"{other_err_pct:.1%}",
        })

    if summary_rows:
        res_df = pd.DataFrame(summary_rows)
        print("\n" + res_df.to_string(index=False))

        # Save summary report
        out_summary_csv = RESULTS_DIR_V2 / "scene_graph_v2_summary.csv"
        res_df.to_csv(out_summary_csv, index=False)
        print(f"\n[OK] Summary table saved to {out_summary_csv}")


if __name__ == "__main__":
    run()
