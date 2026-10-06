import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("Figures for Paper", exist_ok=True)
sns.set_context("paper", font_scale=1.2)

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
    # compute correctness
    combined['accuracy'] = (combined['predicted'] == combined['true_legs']).astype(int)
    return combined

def generate_heatmap(df):
    plt.figure(figsize=(10, 8))
    pivot = df.pivot_table(index='concept', columns='model', values='bias_rate', aggfunc='mean')
    # Sort models roughly by parameter size from left to right
    model_order = ["LLaVA-1.5 (7B/HF)", "LLaVA-1.5 (7B/Oll)", "MiniCPM-V (8B)", "Llama-3.2-V (11B)", "Llama-4-Scout (17B)"]
    # only keep models that exist in pivot
    model_order = [m for m in model_order if m in pivot.columns]
    pivot = pivot[model_order]
    
    # Sort concepts alphabetically 
    pivot = pivot.sort_index()
    
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Bias Rate'})
    plt.title("Bonus Figure A: Concept-Level Bias Heatmap", pad=20)
    plt.ylabel("Animal Concept")
    plt.xlabel("")
    plt.xticks(rotation=35, ha="right")
    
    plt.tight_layout()
    plt.savefig("Figures for Paper/Bonus1_Concept_Heatmap.pdf", dpi=300, format='pdf')
    plt.close()

def generate_anomaly_direction(df):
    plt.figure(figsize=(9, 5))
    
    # Create an anomaly type column
    df['anomaly_type'] = np.where(df['true_legs'] < 4, "Missing Legs (-1)", "Extra Legs (+1)")
    
    sns.barplot(data=df, x='model', y='bias_rate', hue='anomaly_type', palette="coolwarm", capsize=.1)
    
    plt.title("Bonus Figure B: Anomaly Asymmetry (Missing vs Extra)")
    plt.ylabel("Bias Rate (Fallback to Prior)")
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Counterfactual Type")
    
    plt.tight_layout()
    plt.savefig("Figures for Paper/Bonus2_Anomaly_Direction.pdf", dpi=300, format='pdf')
    plt.close()

def generate_scaling_law(df):
    plt.figure(figsize=(7, 5))
    
    # Map model strings to parameter counts
    scale_map = {
        "LLaVA-1.5 (7B/HF)": 7.0,
        "LLaVA-1.5 (7B/Oll)": 7.0,
        "MiniCPM-V (8B)": 8.0,
        "Llama-3.2-V (11B)": 11.0,
        "Llama-4-Scout (17B)": 17.0
    }
    
    overall = df.groupby('model')['bias_rate'].mean().reset_index()
    overall['parameters'] = overall['model'].map(scale_map)
    
    sns.regplot(data=overall, x='parameters', y='bias_rate', 
                scatter_kws={'s': 100, 'color': 'darkblue', 'alpha': 0.7}, 
                line_kws={'color': 'darkred', 'linestyle': '--'})
    
    # Annotate points
    for idx, row in overall.iterrows():
        plt.text(row['parameters'] + 0.3, row['bias_rate'], row['model'].split(' ')[0], fontsize=9)
        
    plt.title("Bonus Figure C: Architecture Scaling Law")
    plt.ylabel("Overall Bias Rate")
    plt.xlabel("Model Size (Billions of Parameters)")
    plt.ylim(0, 1.1)
    plt.xlim(5, 20)
    
    plt.tight_layout()
    plt.savefig("Figures for Paper/Bonus3_Scaling_Law.pdf", dpi=300, format='pdf')
    plt.close()

def run():
    print("Loading mapped parameter datasets...")
    df = load_unified_data()
    df['bias_rate'] = df['is_biased']
    
    print("Rendering Heatmap...")
    generate_heatmap(df)
    
    print("Rendering Anomaly Asymmetry...")
    generate_anomaly_direction(df)
    
    print("Rendering Parameter Scaling Law...")
    generate_scaling_law(df)
    print("All bonus analytical figures saved to 'Figures for Paper/'.")

if __name__ == "__main__":
    run()
