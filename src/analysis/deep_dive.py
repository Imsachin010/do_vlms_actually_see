import pandas as pd
from scipy.stats import spearmanr
import numpy as np

# 1. Visual Distinctiveness Dictionary
visual_salience = {
    'elephant': 1.0, 'pangolin': 0.9, 'deer': 0.8, 'okapi': 0.7,
    'tiger': 0.7, 'leopard': 0.7, 'cheetah': 0.7, 'cat': 0.7,
    'lion': 0.6, 'horse': 0.6, 'fox': 0.6, 'raccoon': 0.6,
    'hyena': 0.5, 'serval': 0.5, 'dog': 0.4, 'quokka': 0.4,
    'binturong': 0.3, 'numbat': 0.3, 'fossa': 0.2
}

def run():
    df = pd.read_csv('results/tables/vlm_test_results.csv')
    
    # === Analysis 1 ===
    # Calculate divergence (LLaVA-1.5 minus Llama-4-Scout bias_rate)
    divergence = df.groupby('concept').apply(lambda x: 
        x[x['model'] == 'LLaVA-1.5']['bias_rate'].mean() - 
        x[x['model'] == 'Llama-4-Scout']['bias_rate'].mean()
    ).reset_index()
    divergence.columns = ['concept', 'divergence']
    
    divergence['salience'] = divergence['concept'].map(visual_salience)
    divergence = divergence.dropna()
    
    rho, p = spearmanr(divergence['salience'], divergence['divergence'])
    print(f"=== Analysis 1: Visual Salience vs Model Divergence ===")
    print(f"Spearman's rho = {rho:.3f}, p = {p:.4f} (n={len(divergence)})\n")
    
    # === Analysis 2: Error Pattern ===
    print("=== Analysis 2: Error Patterns ===")
    # raw prediction data
    llama = pd.read_csv("data/processed/phase2_vlm_results.csv")
    llava = pd.read_csv("data/processed/llava_results.csv")
    llama["model"] = "Llama-4-Scout"
    llava["model"] = "LLaVA-1.5"
    raw = pd.concat([llama, llava])
    
    # Filter only errors
    errors = raw[raw['predicted'] != raw['true_legs']].copy()
    
    def cat_error(row):
        if row['predicted'] == 4: return 'Prior-aligned (4)'
        elif row['predicted'] in [3, 5]: return 'Off-by-one (3 or 5)'
        else: return 'Other/Random'
        
    errors['err_type'] = errors.apply(cat_error, axis=1)
    
    for model in ['LLaVA-1.5', 'Llama-4-Scout']:
        sub = errors[errors['model'] == model]
        vc = sub['err_type'].value_counts(normalize=True) * 100
        print(f"[{model}] Error Distribution (out of {len(sub)} total errors):")
        for k, v in vc.items():
            print(f"  {k}: {v:.1f}%")
        print()

if __name__ == "__main__":
    run()
