"""Compute error-type distribution across all 13 models."""
import pandas as pd
import os

def classify_error(row):
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
        return "prior-aligned"
    elif abs(pred - true) <= 1 and pred != 4:
        return "off-by-one"
    elif 1 <= pred <= 8:
        return "visual-confusion"
    else:
        return "other"

# Load each model
def load_all_models():
    records = []
    
    # 1) Llama-4-Scout local (phase2)
    df = pd.read_csv("data/processed/phase2_vlm_results.csv")
    df["model"] = "Llama-4-Scout"
    records.append(df)
    
    # 2) LLaVA-1.5 local
    df = pd.read_csv("data/processed/llava_results.csv")
    df["model"] = "LLaVA-1.5"
    records.append(df)
    
    # 3-4) Ollama models
    df = pd.read_csv("data/processed/phase3_ollama_results.csv")
    df = df[df["model"] != "moondream"]
    df = df[df["model"] != "llava:7b"]  # skip duplicate
    model_map = {"minicpm-v": "MiniCPM-V", "llama3.2-vision": "Llama-3.2-V"}
    df["model"] = df["model"].map(model_map)
    records.append(df)
    
    # 5-11) Results10
    r10_map = {
        "smolvlm2_results.csv": "SmolVLM2-2.2B",
        "gemma3_4b_results.csv": "Gemma-3-4B",
        "instructblip_results.csv": "InstructBLIP-FlanT5XL",
        "qwen2vl_2b_results.csv": "Qwen2-VL-2B",
        "qwen25_vl_results.csv": "Qwen2.5-VL-7B",
        "llava16_mistral_results.csv": "LLaVA-v1.6-Mistral-7B",
    }
    for fname, model_name in r10_map.items():
        path = os.path.join("Figures for Paper/results10", fname)
        df = pd.read_csv(path)
        df["model"] = model_name
        records.append(df)
    
    # 12-14) HF raw
    hf_models = {
        "Gemma-4-31B": "gemma_4_31b_results.csv",
        "Qwen3-VL-235B-A22B": "qwen3_vl_235b_a22b_results.csv",
        "Qwen3.5-9B": "qwen35_9b_results.csv",
        "Kimi-K2.6": "kimi_k26_results.csv",
    }
    for mn, fn in hf_models.items():
        path = os.path.join("data/raw/results_hf", fn)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["model"] = mn
            records.append(df)
    
    combined = pd.concat(records, ignore_index=True)
    # Dedup
    combined = combined.drop_duplicates(subset=["model", "image", "concept", "background"])
    return combined

combined = load_all_models()
print(f"Total models: {combined['model'].nunique()}")
print(f"Models: {sorted(combined['model'].unique())}")

# Focus on errors: biased samples
errors = combined[combined["is_biased"] == 1].copy()
errors["err_type"] = errors.apply(classify_error, axis=1)

# Per model counts
dist = errors.groupby(["model", "err_type"]).size().unstack(fill_value=0)
# Order columns
col_order = ["prior-aligned", "off-by-one", "visual-confusion", "other"]
for c in col_order:
    if c not in dist.columns:
        dist[c] = 0
dist = dist[col_order]

# Convert to percentages
dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

print("\n=== Error type distribution (% of errors per model) ===")
print(f"{'Model':<30} {'Prior-Aligned':>15} {'Off-by-1':>10} {'Vis-Conf':>10} {'Other':>10} {'Total':>8}")
print("-" * 85)
for model in dist_pct.index:
    row = dist_pct.loc[model]
    total = int(dist.loc[model].sum())
    print(f"{model:<30} {row['prior-aligned']:>13.1f}% {row['off-by-one']:>8.1f}% {row['visual-confusion']:>8.1f}% {row['other']:>8.1f}% {total:>8}")

print("\n=== Raw counts ===")
for model in dist.index:
    row = dist.loc[model]
    total = int(row.sum())
    print(f"{model:<30} pa={int(row['prior-aligned']):>3} ob1={int(row['off-by-one']):>3} vc={int(row['visual-confusion']):>3} oth={int(row['other']):>3} total={total}")
