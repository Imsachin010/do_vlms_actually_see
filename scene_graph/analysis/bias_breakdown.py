"""
Deep-dive analysis of scene graph evaluation results.
Breaks down error types by concept, background, and leg-count.
"""

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("scene_graph/results")
OUTPUT_DIR = Path("scene_graph/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────
MODELS = [
    "Gemma-4-31B",
    "Qwen3.5-9B",
    "Kimi-K2.6",
    "Qwen3-VL-235B-A22B",
    "Llama-4-Scout",
]


def load_sg_results(model_name: str) -> pd.DataFrame:
    slug = model_name.lower().replace("-", "_").replace(".", "")
    csv_path = RESULTS_DIR / f"{slug}_sg_results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["true_legs"] = pd.to_numeric(df["true_legs"], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    return df


def run():
    all_dfs = []
    for model_name in MODELS:
        df = load_sg_results(model_name)
        if df is not None:
            df["model"] = model_name
            all_dfs.append(df)
            print(f"Loaded {model_name}: {len(df)} samples")

    if not all_dfs:
        print("No scene graph results found. Run evaluate_scene_graph.py first.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── 1. Error-type distribution per model ────────────────────────
    error_pivot = combined.groupby(["model", "error_type"]).size().unstack(fill_value=0)
    error_pivot_pct = error_pivot.div(error_pivot.sum(axis=1), axis=0)
    error_pivot.to_csv(OUTPUT_DIR / "sg_error_counts.csv")
    error_pivot_pct.to_csv(OUTPUT_DIR / "sg_error_proportions.csv")
    print("\n=== Error Type Distribution (counts) ===")
    print(error_pivot)
    print("\n=== Error Type Distribution (proportions) ===")
    print(error_pivot_pct.round(3))

    # ── 2. Semantic bias rate by concept ────────────────────────────
    bias_by_concept = (
        combined[combined["error_type"] == "semantic_bias"]
        .groupby(["model", "concept"])
        .size()
        .unstack(fill_value=0)
    )
    bias_by_concept.to_csv(OUTPUT_DIR / "sg_bias_by_concept.csv")
    print("\n=== Semantic Bias Counts by Concept ===")
    print(bias_by_concept)

    # ── 3. Bias rate by true_legs (3 vs 5) ─────────────────────────
    bias_by_legs = (
        combined[combined["error_type"] == "semantic_bias"]
        .groupby(["model", "true_legs"])
        .size()
        .unstack(fill_value=0)
    )
    bias_by_legs.to_csv(OUTPUT_DIR / "sg_bias_by_legs.csv")
    print("\n=== Semantic Bias by Leg Count ===")
    print(bias_by_legs)

    # ── 4. Accuracy by background (plain vs natural) ────────────────
    combined["correct"] = (combined["predicted"] == combined["true_legs"]).astype(int)
    acc_by_bg = combined.groupby(["model", "background"])["correct"].mean().unstack()
    acc_by_bg.to_csv(OUTPUT_DIR / "sg_accuracy_by_background.csv")
    print("\n=== Accuracy by Background ===")
    print(acc_by_bg.round(3))

    # ── 5. Overall accuracy summary ─────────────────────────────────
    overall_acc = combined.groupby("model")["correct"].mean()
    overall_acc.to_csv(OUTPUT_DIR / "sg_overall_accuracy.csv")
    print("\n=== Overall Accuracy ===")
    print(overall_acc.round(3))

    # ── 6. Where models fail completely (parse_failed / api_error) ──
    failures = combined[combined["status"] != "ok"]
    fail_counts = failures.groupby(["model", "status"]).size().unstack(fill_value=0)
    fail_counts.to_csv(OUTPUT_DIR / "sg_failure_counts.csv")
    print("\n=== Failures by Model ===")
    print(fail_counts)

    print(f"\n\nAll analysis tables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
