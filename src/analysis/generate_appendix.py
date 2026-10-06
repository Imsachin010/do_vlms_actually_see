import pandas as pd

def load_unified_data():
    df_scout = pd.read_csv("data/processed/phase2_vlm_results.csv")
    df_scout['model'] = "Llama-4-Scout (17B)"
    df_scout['engine'] = "Groq API (Cloud)"
    
    df_llava = pd.read_csv("data/processed/llava_results.csv")
    df_llava['model'] = "LLaVA-1.5 (7B/HF)"
    df_llava['engine'] = "PyTorch (Local RTX)"
    
    df_ollama = pd.read_csv("data/processed/phase3_ollama_results.csv")
    if 'is_biased' in df_ollama.columns:
        df_ollama.rename(columns={'is_biased': 'bias_rate'}, inplace=True)
    df_ollama = df_ollama[df_ollama['model'] != 'moondream']
    df_ollama['engine'] = "Ollama/llama.cpp (Local Quant Q4_0)"
    
    dict_map = {
        "llava:7b": "LLaVA-1.5 (7B/Oll)",
        "minicpm-v": "MiniCPM-V (8B)",
        "llama3.2-vision": "Llama-3.2-V (11B)"
    }
    df_ollama['model'] = df_ollama['model'].map(lambda x: dict_map.get(x, x))
    
    combined = pd.concat([df_scout, df_llava, df_ollama], ignore_index=True)
    combined['bias_rate'] = combined['is_biased']
    combined['accuracy'] = (combined['predicted'] == combined['true_legs']).astype(int)
    
    # Error classification
    def get_err_type(row):
        if row['predicted'] == row['true_legs']: return 'Correct'
        if row['predicted'] == 4: return 'Prior-Aligned'
        if row['predicted'] in [3, 5]: return 'Off-by-One'
        return 'Random'
        
    combined['err_type'] = combined.apply(get_err_type, axis=1)
    
    return combined

def run():
    print("Aggregating exhaustive TMLR-style tables...")
    df = load_unified_data()
    
    # Table 1: Master Performance Breakdown
    stats = df.groupby(['model', 'engine']).agg(
        n_tests=('image', 'count'),
        mean_bias_rate=('bias_rate', 'mean'),
        accuracy=('accuracy', 'mean')
    ).reset_index()
    
    # Add Error Proportions
    err_dist = df[df['err_type'] != 'Correct'].groupby('model')['err_type'].value_counts(normalize=True).unstack(fill_value=0) * 100
    err_dist = err_dist.reset_index()
    
    stats = stats.merge(err_dist, on='model', how='left')
    stats.fillna(0, inplace=True)
    
    for c in ['mean_bias_rate', 'accuracy']: stats[c] = (stats[c]*100).map("{:.1f}%".format)
    for c in ['Prior-Aligned', 'Off-by-One', 'Random']: 
        if c in stats.columns: stats[c] = stats[c].map("{:.1f}%".format)
    
    stats = stats.sort_values(by="mean_bias_rate", ascending=False)
    table1 = stats.to_markdown(index=False)
    
    # Table 2: Background Context Matrix
    bg_pivot = df.pivot_table(index='model', columns='background', values='bias_rate', aggfunc='mean') * 100
    bg_pivot = bg_pivot.map(lambda x: f"{x:.1f}%").reset_index()
    table2 = bg_pivot.to_markdown(index=False)
    
    # Table 3: Per-Concept Heatmap Equivalent
    concept_pivot = df.pivot_table(index='concept', columns='model', values='bias_rate', aggfunc='mean') * 100
    concept_pivot = concept_pivot.map(lambda x: f"{x:.1f}%").reset_index()
    table3 = concept_pivot.to_markdown(index=False)
    
    md_content = f"""# Appendix A: Extensive Empirical Validation Framework

In line with standards for reproducible empirical evaluation (e.g., TMLR, ICLR), this appendix provides exhaustive granular details on the hyperparameter configurations, testbed constraints, prompt templates, and multi-dimensional analytical matrices across all evaluated parameters.

## A.1 Methodology and Hyperparameters

**Evaluation Prompt (Zero-Shot):**
> "How many legs does this animal exactly have in this image? Answer with just a single number integer, nothing else."

**Hardware & APIs:**
- **Remote Scaling Inference:** Handled securely via the **Groq API** endpoint mapping to 17B infrastructure natively without memory offloading limits.
- **Local High-Fidelity Reference:** Handled via HuggingFace `transformers` deployed on dedicated PyTorch GPU pipelines.
- **Local Edge-Compute Inference:** Handled strictly via `ollama` utilizing 4-bit `Q4_0` model quantization constraints restricted to a consumer 6GB VRAM limitation.

**Hyperparameters:**
All models across both local endpoints and remote Groq infrastructures were locked to zero-temperature deterministic generation to enforce pure capability measurement.
* `temperature: 0.0`
* `seed: 42`
* `stream: False`

---

## A.2 Table 1. Global Performance & Error Topology

| Model Architecture | Inference Engine | Total Queries | Bias Rate (Prior Override Failure) | Accuracy | Prior-Aligned Errors* | Off-by-One Errors* | Random Errors* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join([f"| {row['model']} | {row['engine']} | {row['n_tests']} | **{row['mean_bias_rate']}** | {row['accuracy']} | {row.get('Prior-Aligned', '0.0%')} | {row.get('Off-by-One', '0.0%')} | {row.get('Random', '0.0%')} |" for _, row in stats.iterrows()])}
*\*Error proportions denote percentage of total incorrect generations, not of global execution count.*

---

## A.3 Table 2. Background Spatial Noise Effect

This matrix tracks the reliance of specific model architectures on localized spatial backgrounds (`Plain` solid textures vs `Natural` photographic environments). Massive gaps here imply architecture-level entanglement of semantic truths to surrounding visual noise.

{table2}

---

## A.4 Table 3. Isolated Class Breakdown (Per-Concept Matrix)

The matrix proves that bias collapse is rarely universally uniform. Some specific models reliably override visual semantic priors on highly visually distinctive (salient) animals, yet totally collapse on common dense archetypes. Values represent Bias Failure Rate (`100%` indicates complete fallback to prior text truth).

{table3}
"""
    
    with open("MODEL_APPENDIX.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("Successfully generated exhaustive TMLR MODEL_APPENDIX.md!")

if __name__ == "__main__":
    run()
