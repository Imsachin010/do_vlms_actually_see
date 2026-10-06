"""Check Gemma-4-31B pred=3 and pred=5 errors."""
import pandas as pd

df = pd.read_csv("data/raw/results_hf/gemma_4_31b_results.csv")
df = df.dropna(subset=["predicted"])

print("=== Gemma-4-31B pred=3 cases ===")
pred3 = df[df["predicted"].astype(float) == 3]
for _, r in pred3.iterrows():
    print(f"  {r['concept']:<15} true={r['true_legs']:.0f} is_biased={r['is_biased']} background={r.get('background','N/A')}")

print("\n=== Gemma-4-31B pred=5 cases ===")
pred5 = df[df["predicted"].astype(float) == 5]
for _, r in pred5.iterrows():
    print(f"  {r['concept']:<15} true={r['true_legs']:.0f} is_biased={r['is_biased']} background={r.get('background','N/A')}")

print("\n=== Gemma-4-31B ALL pred values ===")
print(df["predicted"].value_counts().sort_index())
