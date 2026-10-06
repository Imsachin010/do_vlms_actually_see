"""Recalculate error types using the original definition: off-by-one = pred in [3, 5]."""
import pandas as pd
import os

def classify_error_original(row):
    """Original definition from generate_5model_paper.py"""
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
    elif pred in [3, 5]:
        return "off-by-one"  # ±1 from canonical 4
    elif 1 <= pred <= 8:
        return "visual-confusion"
    else:
        return "other"

def classify_error_groundtruth(row):
    """Alternative: off-by-one = ±1 from ground truth"""
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

def load_data():
    records = []
    # 1) Llama-4-Scout local
    df = pd.read_csv("data/processed/phase2_vlm_results.csv")
    df["model"] = "Llama-4-Scout"
    records.append(df)
    # 2) LLaVA-1.5 local
    df = pd.read_csv("data/processed/llava_results.csv")
    df["model"] = "LLaVA-1.5"
    records.append(df)
    # 3-4) Ollama
    df = pd.read_csv("data/processed/phase3_ollama_results.csv")
    df = df[df["model"] != "moondream"]
    df = df[df["model"] != "llava:7b"]
    m = {"minicpm-v": "MiniCPM-V", "llama3.2-vision": "Llama-3.2-V"}
    df["model"] = df["model"].map(m)
    records.append(df)
    # 5-11) Results10
    r10 = {"smolvlm2_results.csv": "SmolVLM2-2.2B", "gemma3_4b_results.csv": "Gemma-3-4B",
           "instructblip_results.csv": "InstructBLIP-FlanT5XL", "qwen2vl_2b_results.csv": "Qwen2-VL-2B",
           "qwen25_vl_results.csv": "Qwen2.5-VL-7B", "llava16_mistral_results.csv": "LLaVA-v1.6-Mistral-7B"}
    for fn, mn in r10.items():
        records.append(pd.read_csv(os.path.join("Figures for Paper/results10", fn)).assign(model=mn))
    # 12-14) HF
    hf = {"Gemma-4-31B": "gemma_4_31b_results.csv", "Qwen3-VL-235B-A22B": "qwen3_vl_235b_a22b_results.csv",
          "Qwen3.5-9B": "qwen35_9b_results.csv", "Kimi-K2.6": "kimi_k26_results.csv"}
    for mn, fn in hf.items():
        path = os.path.join("data/raw/results_hf", fn)
        if os.path.exists(path):
            records.append(pd.read_csv(path).assign(model=mn))
    combined = pd.concat(records, ignore_index=True)
    combined = combined.drop_duplicates(subset=["model", "image", "concept", "background"])
    return combined

df = load_data()
errors = df[df["predicted"].astype(float) != df["true_legs"].astype(float)].copy()

# Both definitions
for name, classify_fn in [("Original (canonical-based)", classify_error_original), 
                           ("GT-based", classify_error_groundtruth)]:
    errors["err_type"] = errors.apply(classify_fn, axis=1)
    dist = errors.groupby(["model", "err_type"]).size().unstack(fill_value=0)
    col_order = ["prior-aligned", "off-by-one", "visual-confusion", "other"]
    for c in col_order:
        if c not in dist.columns:
            dist[c] = 0
    dist = dist[col_order]
    totals = dist.sum(axis=1)
    dist_pct = dist.div(totals, axis=0) * 100
    
    print(f"\n=== {name} ===")
    for model in dist_pct.index:
        row = dist_pct.loc[model]
        raw = dist.loc[model]
        print(f"{model:<30} pa={raw['prior-aligned']:>3} ob1={raw['off-by-one']:>3} vc={raw['visual-confusion']:>3} oth={raw['other']:>3} total={int(totals[model]):>3}  "
              f"(pa={row['prior-aligned']:>5.1f}% ob1={row['off-by-one']:>5.1f}% vc={row['visual-confusion']:>5.1f}% oth={row['other']:>5.1f}%)")
