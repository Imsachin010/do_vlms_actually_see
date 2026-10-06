import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("Figures for Paper", exist_ok=True)
sns.set_context("paper", font_scale=1.0)
sns.set_style("whitegrid")

def load_unified_data():
    df_scout = pd.read_csv("data/processed/phase2_vlm_results.csv")
    df_scout['model'] = "Llama-4-Scout (17B)"
    
    df_llava = pd.read_csv("data/processed/llava_results.csv")
    df_llava['model'] = "LLaVA-1.5 (7B/HF)"
    
    df_ollama = pd.read_csv("data/processed/phase3_ollama_results.csv")
    if 'is_biased' in df_ollama.columns:
        df_ollama.rename(columns={'is_biased': 'bias_rate'}, inplace=True)
    df_ollama = df_ollama[df_ollama['model'] != 'moondream']
    
    dict_map = {
        "llava:7b": "LLaVA-1.5 (7B/Oll)",
        "minicpm-v": "MiniCPM-V (8B)",
        "llama3.2-vision": "Llama-3.2-V (11B)"
    }
    df_ollama['model'] = df_ollama['model'].map(lambda x: dict_map.get(x, x))
    
    combined = pd.concat([df_scout, df_llava, df_ollama], ignore_index=True)
    combined['bias_rate'] = combined['is_biased']
    return combined

def run():
    print("Loading data for concept panel grid...")
    df = load_unified_data()
    
    # Calculate mean bias rate per concept for ordering
    concept_order = df.groupby('concept')['bias_rate'].mean().sort_values(ascending=False).index

    print("Rendering FacetGrid Panels...")
    g = sns.catplot(
        data=df, x='concept', y='bias_rate', col='model', 
        kind='bar', col_wrap=3, height=4, aspect=1.2, 
        palette="viridis", order=concept_order, errorbar=None
    )
    
    g.set_axis_labels("", "Bias Rate (Fallback to Prior)")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1.05))
    
    # Rotate x labels for readability on every facet
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    plt.suptitle("Bonus Figure D: Bias Distribution Across Animal Classes per Model", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures for Paper/Bonus4_Concept_Panels.pdf", dpi=300, format='pdf')
    plt.close()
    print("Saved 'Bonus4_Concept_Panels.pdf' successfully.")

if __name__ == "__main__":
    run()
