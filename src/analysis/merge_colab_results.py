"""
merge_colab_results.py
─────────────────────
After running main.ipynb on Colab and downloading the results/ folder from
Google Drive, place the CSVs into:
    data/processed/colab_results/

Then run:
    python -m src.analysis.merge_colab_results

This produces:
    data/processed/master_10model_results.csv
"""

import os
import glob
import pandas as pd

INPUT_DIR  = "data/processed/colab_results"
OUTPUT_CSV = "data/processed/master_10model_results.csv"

# Friendly display names matching what the notebook wrote
DISPLAY_NAMES = {
    "smolvlm2_results.csv":       "SmolVLM2-2.2B",
    "llava15_results.csv":        "LLaVA-1.5-7B",
    "llava16_mistral_results.csv":"LLaVA-v1.6-Mistral-7B",
    "phi35_vision_results.csv":   "Phi-3.5-Vision-4.2B",
    "internvl25_4b_results.csv":  "InternVL2_5-4B",
    "gemma3_4b_results.csv":      "Gemma-3-4B",
    "qwen25_vl_results.csv":      "Qwen2.5-VL-7B",
    "internvl2_8b_results.csv":   "InternVL2-8B",
    "idefics3_results.csv":       "Idefics3-8B-Llama3",
    "llama4scout_results.csv":    "Llama-4-Scout-17B",
}

def run():
    os.makedirs(INPUT_DIR, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    
    if not csv_files:
        print(f"No CSVs found in {INPUT_DIR}/")
        print("Please download your Google Drive results/ folder and place the CSVs there.")
        return
    
    frames = []
    print(f"{'CSV File':<45} {'Rows':>6}  {'Bias Rate':>10}")
    print("─" * 65)
    
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath)
        display = DISPLAY_NAMES.get(fname, df['model'].iloc[0] if 'model' in df.columns else fname)
        df['model'] = display
        bias_rate = df['is_biased'].mean() * 100
        print(f"{fname:<45} {len(df):>6}  {bias_rate:>9.1f}%")
        frames.append(df)
    
    master = pd.concat(frames, ignore_index=True)
    master.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Master dataset saved → {OUTPUT_CSV}")
    print(f"   Total rows : {len(master)}")
    print(f"   Models     : {master['model'].nunique()}")
    print(f"\n📊 Overall Bias Rate by Model:")
    print(master.groupby('model')['is_biased'].mean().mul(100).round(1).to_string())
    print("\nRun `src/analysis/generate_5model_paper.py` after pointing it to the master CSV.")

if __name__ == "__main__":
    run()
