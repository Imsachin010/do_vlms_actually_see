"""Explore error types for Llama-4-Scout to design the stacked bar chart."""
import pandas as pd
import numpy as np

def classify_error(row):
    """Classify an error into one of four categories."""
    pred = row.get("predicted", None)
    true = row.get("true_legs", None)
    
    if pd.isna(pred) or pd.isna(true):
        return "other"
    
    try:
        pred = float(pred)
        true = float(true)
    except (ValueError, TypeError):
        return "other"
    
    if pred == 4:
        return "prior-aligned"  # canonical "4"
    elif abs(pred - true) <= 1 and pred != 4:
        return "off-by-one"  # ±1 from ground truth but not 4
    elif 1 <= pred <= 8:
        return "visual-confusion"  # non-canonical wrong answer not off-by-one
    else:
        return "other"  # out of range

# Load data for Llama-4-Scout local
df = pd.read_csv("data/processed/phase2_vlm_results.csv")
sub = df[df["model"].str.strip() == "llama4-scout"].copy()
errors = sub[sub["is_biased"] == 1].copy()
errors["err_type"] = errors.apply(classify_error, axis=1)
print("=== Llama-4-Scout (local, phase2) error breakdown ===")
print(errors["err_type"].value_counts())
print(f"Total errors: {len(errors)}")

# Llama-4-Scout HF (raw) — show what the "other" category predictions look like
df2 = pd.read_csv("data/raw/results_hf/llama_4_scout_results.csv")
errors2 = df2[df2["is_biased"] == 1].copy()
errors2["err_type"] = errors2.apply(classify_error, axis=1)
print("\n=== Llama-4-Scout (HF raw) error breakdown ===")
print(errors2["err_type"].value_counts())
print(f"Total errors: {len(errors2)}")

# Show what "other" predictions look like
other = errors2[errors2["err_type"] == "other"]
print("\n=== Llama-4-Scout 'other' errors ===")
for _, row in other.iterrows():
    print(f"  {row['concept']:<15} true={row['true_legs']}  pred='{row['predicted']}'  status={row['status']}")

# Show off-by-one and visual confusion
for cat in ["off-by-one", "visual-confusion"]:
    sub = errors2[errors2["err_type"] == cat]
    print(f"\n=== Llama-4-Scout '{cat}' errors ===")
    for _, row in sub.iterrows():
        print(f"  {row['concept']:<15} true={row['true_legs']}  pred={row['predicted']}")

# Gemma-4-31B
df3 = pd.read_csv("data/raw/results_hf/gemma_4_31b_results.csv")
errors3 = df3[df3["is_biased"] == 1].copy()
errors3["err_type"] = errors3.apply(classify_error, axis=1)
print("\n=== Gemma-4-31B error breakdown ===")
print(errors3["err_type"].value_counts())
print(f"Total errors: {len(errors3)}")

# Qwen3.5-9B
df4 = pd.read_csv("data/raw/results_hf/qwen35_9b_results.csv")
errors4 = df4[df4["is_biased"] == 1].copy()
errors4["err_type"] = errors4.apply(classify_error, axis=1)
print("\n=== Qwen3.5-9B error breakdown ===")
print(errors4["err_type"].value_counts())
print(f"Total errors: {len(errors4)}")
