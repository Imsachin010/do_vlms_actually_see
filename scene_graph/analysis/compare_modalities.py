"""
Compare raw-image vs scene-graph performance for the same models.
Isolates:
  - Perception error:   model got image wrong but scene graph right
  - Reasoning error:    model got scene graph wrong (no visual ambiguity)
  - Semantic bias:      model answered "4" despite counterfactual info
"""

import json
import os
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("scene_graph/results")
RAW_RESULTS_DIR = Path("data/raw/results_hf")
OUTPUT_DIR = Path("scene_graph/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_results(model_name: str) -> pd.DataFrame:
    """
    Load raw image-inference results from the main experiment.
    We match on model display name -> slug.
    """
    slug = model_name.lower().replace("-", "_").replace(".", "")
    csv_path = RAW_RESULTS_DIR / f"{slug}_results.csv"
    if not csv_path.exists():
        print(f"  [WARN] No raw results found for {model_name} at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    # Ensure consistent types
    df["true_legs"] = pd.to_numeric(df["true_legs"], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    # Extract image_id from filename (strip .png/.jpg)
    df["image_id"] = df["image"].apply(
        lambda x: str(x).rsplit(".", 1)[0] if pd.notna(x) else x
    )
    return df


def load_sg_results(model_name: str) -> pd.DataFrame:
    slug = model_name.lower().replace("-", "_").replace(".", "")
    csv_path = RESULTS_DIR / f"{slug}_sg_results.csv"
    if not csv_path.exists():
        print(f"  [WARN] No SG results found for {model_name} at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["true_legs"] = pd.to_numeric(df["true_legs"], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    return df


def compute_metrics(df_raw: pd.DataFrame, df_sg: pd.DataFrame, model_name: str) -> dict:
    """
    For each image, compare raw vs SG performance.
    """
    merged = df_raw[["image_id", "true_legs", "predicted", "background", "concept"]].merge(
        df_sg[["image_id", "true_legs", "predicted", "error_type"]],
        on="image_id",
        suffixes=("_raw", "_sg"),
        how="inner",
    )

    # Only consider entries where both predictions are valid
    merged = merged.dropna(subset=["predicted_raw", "predicted_sg"])
    merged["predicted_raw"] = merged["predicted_raw"].astype(int)
    merged["predicted_sg"] = merged["predicted_sg"].astype(int)
    merged["true_legs"] = merged["true_legs_raw"].astype(int)

    # Classification
    merged["correct_raw"] = (merged["predicted_raw"] == merged["true_legs"]).astype(int)
    merged["correct_sg"] = (merged["predicted_sg"] == merged["true_legs"]).astype(int)
    merged["bias_raw"] = (merged["predicted_raw"] == 4).astype(int)
    merged["bias_sg"] = (merged["predicted_sg"] == 4).astype(int)

    # Error categories
    merged["perception_error"] = (
        (~merged["correct_raw"].astype(bool)) & merged["correct_sg"].astype(bool)
    ).astype(int)
    merged["reasoning_error"] = (
        (~merged["correct_sg"].astype(bool))
    ).astype(int)
    merged["both_wrong"] = (
        (~merged["correct_raw"].astype(bool)) & (~merged["correct_sg"].astype(bool))
    ).astype(int)

    n = len(merged)

    metrics = {
        "model": model_name,
        "total_samples": n,
        "raw_accuracy": merged["correct_raw"].mean(),
        "sg_accuracy": merged["correct_sg"].mean(),
        "accuracy_gain": merged["correct_sg"].mean() - merged["correct_raw"].mean(),
        "raw_bias_rate": merged["bias_raw"].mean(),
        "sg_bias_rate": merged["bias_sg"].mean(),
        "bias_reduction": merged["bias_raw"].mean() - merged["bias_sg"].mean(),
        "perception_error_rate": merged["perception_error"].mean(),
        "reasoning_error_rate": merged["reasoning_error"].mean(),
        "both_wrong_rate": merged["both_wrong"].mean(),
        "sg_perfect_count": merged["correct_sg"].sum(),
    }

    return metrics, merged


# ── Models to compare (same as the HF tests) ───────────────────────────
MODELS = [
    "Gemma-4-31B",
    "Qwen3.5-9B",
    "Kimi-K2.6",
    "Qwen3-VL-235B-A22B",
    "Llama-4-Scout",
]


def run():
    all_metrics = []
    all_comparisons = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Comparing: {model_name}")
        print(f"{'='*60}")

        df_raw = load_raw_results(model_name)
        df_sg = load_sg_results(model_name)

        if df_raw is None or df_sg is None:
            print(f"  Skipping {model_name} (missing data)")
            continue

        print(f"  Raw images: {len(df_raw)} samples")
        print(f"  Scene graphs: {len(df_sg)} samples")

        metrics, comparison_df = compute_metrics(df_raw, df_sg, model_name)
        all_metrics.append(metrics)
        all_comparisons.append(comparison_df)

        print(f"  Raw accuracy:      {metrics['raw_accuracy']:.1%}")
        print(f"  SG accuracy:       {metrics['sg_accuracy']:.1%}")
        print(f"  Accuracy gain:     {metrics['accuracy_gain']:+.1%}")
        print(f"  Raw bias rate:     {metrics['raw_bias_rate']:.1%}")
        print(f"  SG bias rate:      {metrics['sg_bias_rate']:.1%}")
        print(f"  Bias reduction:    {metrics['bias_reduction']:+.1%}")
        print(f"  Perception errors: {metrics['perception_error_rate']:.1%}")
        print(f"  Reasoning errors:  {metrics['reasoning_error_rate']:.1%}")

    # ── Save aggregated metrics ─────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUTPUT_DIR / "modality_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved aggregated metrics -> {metrics_path}")

    # ── Save full comparison ────────────────────────────────────────
    if all_comparisons:
        full_df = pd.concat(all_comparisons, ignore_index=True)
        full_path = OUTPUT_DIR / "modality_comparison_detail.csv"
        full_df.to_csv(full_path, index=False)
        print(f"Saved detailed comparison -> {full_path}")

    # ── Print summary table ─────────────────────────────────────────
    print("\n\n" + "="*60)
    print("SUMMARY: Raw Vision vs Scene Graph Grounding")
    print("="*60)
    summary_cols = [
        "model", "raw_accuracy", "sg_accuracy", "accuracy_gain",
        "raw_bias_rate", "sg_bias_rate", "bias_reduction",
        "perception_error_rate", "reasoning_error_rate",
    ]
    print(metrics_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    run()
