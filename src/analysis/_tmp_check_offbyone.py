"""
Check off-by-one errors: predicted ±1 from ground truth.
"""
import pandas as pd

# Llama-4-Scout local
df = pd.read_csv("data/processed/phase2_vlm_results.csv")
sub = df[df["model"].str.strip() == "llama4-scout"].copy()
errors = sub[sub["predicted"].astype(float) != sub["true_legs"].astype(float)].copy()

print("=== Llama-4-Scout all errors with GT-based off-by-one ===")
for _, r in errors.iterrows():
    pred = float(r["predicted"])
    true = float(r["true_legs"])
    diff = abs(pred - true)
    if pred == 4:
        cat = "prior-aligned"
    elif diff <= 1:
        cat = "off-by-one"
    elif 1 <= pred <= 8:
        cat = "visual-confusion"
    else:
        cat = "other"
    print(f"{r['concept']:<15} true={true:.0f} pred={pred:.0f} diff={diff:.0f} bg={r['background']:<10} -> {cat}")

# Count
def classify_gt(row):
    pred = float(row["predicted"])
    true = float(row["true_legs"])
    if pred == 4:
        return "prior-aligned"
    elif abs(pred - true) <= 1:
        return "off-by-one"
    elif 1 <= pred <= 8:
        return "visual-confusion"
    else:
        return "other"

errors["err_type"] = errors.apply(classify_gt, axis=1)
print("\nError type distribution:")
print(errors["err_type"].value_counts())
total = len(errors)
for et, cnt in errors["err_type"].value_counts().items():
    print(f"  {et:<20}: {cnt}/{total} = {cnt/total*100:.1f}%")

# Now check if Llama-4-Scout HF raw has any
print("\n=== Llama-4-Scout HF raw errors ===")
df2 = pd.read_csv("data/raw/results_hf/llama_4_scout_results.csv")
sub2 = df2.dropna(subset=["predicted"])
errors2 = sub2[sub2["predicted"].astype(float) != sub2["true_legs"].astype(float)].copy()
errors2["err_type"] = errors2.apply(classify_gt, axis=1)
print(errors2["err_type"].value_counts())
total2 = len(errors2)
for et, cnt in errors2["err_type"].value_counts().items():
    if et == "off-by-one":
        print(f"  {et:<20}: {cnt}/{total2} = {cnt/total2*100:.1f}%")
        for _, r in errors2[errors2["err_type"] == "off-by-one"].iterrows():
            print(f"    {r['concept']:<15} true={r['true_legs']:.0f} pred={float(r['predicted']):.0f}")
