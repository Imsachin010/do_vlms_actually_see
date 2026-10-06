import pandas as pd
from scipy.stats import spearmanr
import numpy as np

# Load your results
df = pd.read_csv('results/tables/vlm_test_results.csv')  

for model in ['LLaVA-1.5', 'Llama-4-Scout']:
    for bg in ['plain', 'natural']:
        subset = df[(df['model'] == model) & (df['background'] == bg)]
        
        # Compute correlation
        rho, p = spearmanr(subset['prior_strength'], subset['bias_rate'])
        
        print(f"{model} - {bg}:")
        print(f"  ρ = {rho:.3f}, p = {p:.4f}")
        print(f"  Sample size: n = {len(subset)}")
        print()
