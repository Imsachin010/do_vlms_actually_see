"""
multiple_regression_bias.py
────────────────────────────
Runs OLS multiple regression on Llama-4-Scout bias rates.
Tests:
  1) bias ~ freq + sep             (partial effects)
  2) bias ~ freq + sep + freq*sep  (interaction term)
"""
import pandas as pd
import statsmodels.api as sm
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = str(BASE_DIR / "results" / "salience_analysis" / "concept_salience_metrics.csv")
OUT_DIR  = str(BASE_DIR / "results" / "salience_analysis")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Needs: bias_rate, concept_frequency_score, clip_separability_score
    df = df.dropna(subset=['bias_rate', 'concept_frequency_score', 'clip_separability_score'])
    
    y = df['bias_rate']
    
    # ── Model 1: Main Effects Only ──────────────────────────────────────────
    X1 = df[['concept_frequency_score', 'clip_separability_score']]
    X1 = sm.add_constant(X1)
    
    model1 = sm.OLS(y, X1).fit()
    
    print("\n" + "="*80)
    print("MODEL 1: Partial Effects (bias ~ frequency + separability)")
    print("="*80)
    print(model1.summary())
    
    # ── Model 2: Interaction Term ───────────────────────────────────────────
    X2 = df[['concept_frequency_score', 'clip_separability_score']].copy()
    X2['interaction'] = X2['concept_frequency_score'] * X2['clip_separability_score']
    X2 = sm.add_constant(X2)
    
    model2 = sm.OLS(y, X2).fit()
    
    print("\n\n" + "="*80)
    print("MODEL 2: Interaction Term (bias ~ frequency + separability + interaction)")
    print("="*80)
    print(model2.summary())

    # Save summary strings to text file for reference
    out_txt = os.path.join(OUT_DIR, "regression_results.txt")
    with open(out_txt, "w") as f:
        f.write("MODEL 1: Partial Effects (bias ~ frequency + separability)\n")
        f.write(model1.summary().as_text() + "\n\n")
        f.write("========================================================\n\n")
        f.write("MODEL 2: Interaction Term (bias ~ frequency + separability + interaction)\n")
        f.write(model2.summary().as_text() + "\n")
        
    print(f"\n✅ Results saved to {out_txt}")

if __name__ == "__main__":
    main()
