"""Temporary script: compute bias rates across all models."""
import pandas as pd
import os

all_data = []

# Phase 2
df = pd.read_csv("data/processed/phase2_vlm_results.csv")
df["model"] = df["model"].str.strip()
all_data.append(df)

# LLaVA-1.5 local
df = pd.read_csv("data/processed/llava_results.csv")
df["model"] = "llava-1.5-7b-ollama"
all_data.append(df)

# Phase 3
df = pd.read_csv("data/processed/phase3_ollama_results.csv")
df = df[df["model"] != "moondream"].copy()
m = {"llava:7b": "LLaVA-1.5 (O)", "minicpm-v": "MiniCPM-V", "llama3.2-vision": "Llama-3.2-V"}
df["model"] = df["model"].map(lambda x: m.get(x, x))
all_data.append(df)

# Results10
for f in os.listdir("Figures for Paper/results10"):
    if f.endswith(".csv") and f != "plots":
        df = pd.read_csv(os.path.join("Figures for Paper/results10", f))
        if "model" in df.columns:
            all_data.append(df)

# HF raw
hf_files = {
    "Gemma-4-31B": "gemma_4_31b_results.csv",
    "Llama-4-Scout": "llama_4_scout_results.csv",
    "Qwen3-VL-235B-A22B": "qwen3_vl_235b_a22b_results.csv",
    "Qwen3.5-9B": "qwen35_9b_results.csv",
    "Kimi-K2.6": "kimi_k26_results.csv",
}
for mn, fn in hf_files.items():
    p = os.path.join("data/raw/results_hf", fn)
    if os.path.exists(p):
        df = pd.read_csv(p)
        if "is_biased" in df.columns:
            df["model"] = mn
            all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
combined = combined.drop_duplicates(subset=["model", "image", "concept", "background"])

# Fix model name: phase2 llama4-scout -> Llama-4-Scout
combined["model"] = combined["model"].replace({
    "llama4-scout": "Llama-4-Scout",
    "llava-1.5-7b-ollama": "LLaVA-1.5",
})

if "is_biased" in combined.columns:
    bias = combined.groupby("model")["is_biased"].mean().reset_index()
    bias = bias.sort_values("is_biased", ascending=False)
    n_total = combined.groupby("model")["is_biased"].count()
    n_biased = combined.groupby("model")["is_biased"].sum()
    bias["n"] = bias["model"].map(n_total)
    bias["n_biased"] = bias["model"].map(n_biased)
    
    print(f"Total unique models: {len(bias)}")
    print()
    
    for _, row in bias.iterrows():
        print(f"{row['model']:<30} bias={row['is_biased']:.1%}  ({int(row['n_biased'])}/{int(row['n'])})")
