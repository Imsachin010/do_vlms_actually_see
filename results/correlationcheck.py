import pandas as pd

wiki = pd.read_csv("data/processed/wikipedia_scores.csv")
clip = pd.read_csv("data/processed/clip_scores.csv")
model = pd.read_csv("data/processed/model_confidence_scores.csv")

df = wiki.merge(clip, on="concept").merge(model, on="concept")

# Correlation checks
corr_model = df["pageviews_norm"].corr(df["model_confidence"], method="spearman")
corr_clip = df["pageviews_norm"].corr(df["clip_zscore"], method="spearman")

print("Wiki vs Model:", corr_model)
print("Wiki vs CLIP:", corr_clip)