import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

os.makedirs("Figures for Paper", exist_ok=True)

# --- Premium Publication Styling ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Professional Color Palettes
PRIMARY_COLOR = "#2C3E50" # Dark Blue-Gray
BACKGROUND_PALETTE = ["#3498DB", "#E74C3C"] # Blue and Red for Natural/Plain
ACCENT_PALETTE = sns.color_palette("viridis", as_cmap=False)

def load_unified_data():
    all_dfs = []
    
    # 1. Llama-4-Scout (Phase 2)
    if os.path.exists("data/processed/phase2_vlm_results.csv"):
        df_scout = pd.read_csv("data/processed/phase2_vlm_results.csv")
        df_scout['model'] = "Llama-4-Scout"
        all_dfs.append(df_scout)
    
    # 2. LLaVA-1.5 (Phase 2)
    if os.path.exists("data/processed/llava_results.csv"):
        df_llava = pd.read_csv("data/processed/llava_results.csv")
        df_llava['model'] = "LLaVA-1.5"
        all_dfs.append(df_llava)
    
    # 3. Phase 3 (Ollama)
    if os.path.exists("data/processed/phase3_ollama_results.csv"):
        df_ollama = pd.read_csv("data/processed/phase3_ollama_results.csv")
        df_ollama = df_ollama[df_ollama['model'] != 'moondream'].copy() # remove debugging artifact
        dict_map = {
            "llava:7b": "LLaVA-1.5 (O)",
            "minicpm-v": "MiniCPM-V",
            "llama3.2-vision": "Llama-3.2-V"
        }
        df_ollama['model'] = df_ollama['model'].map(lambda x: dict_map.get(x, x))
        all_dfs.append(df_ollama)
        
    # 4. Results10 (Additional New Models)
    results10_dir = "Figures for Paper/results10"
    if os.path.exists(results10_dir):
        for f in os.listdir(results10_dir):
            if f.endswith(".csv"):
                rdf = pd.read_csv(os.path.join(results10_dir, f))
                # Ensure model name is clean
                if "model" in rdf.columns:
                    all_dfs.append(rdf)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    # Deduplicate based on model, image, concept to avoid double-counting overlapping sets
    combined = combined.drop_duplicates(subset=['model', 'image', 'concept', 'background'])
    
    # Ensure critical columns exist
    if 'predicted' in combined.columns and 'true_legs' in combined.columns:
        combined['error'] = (combined['predicted'] - combined['true_legs']).abs()
    
    return combined

def generate_figure1_overall(df):
    overall = df.groupby('model')['bias_rate'].mean().reset_index()
    overall = overall.sort_values(by="bias_rate", ascending=False)
    
    plt.figure(figsize=(10, 6.5))
    ax = sns.barplot(
        data=overall, 
        x='model', 
        y='bias_rate', 
        palette='viridis',
        hue='model',
        legend=False,
        edgecolor='black',
        linewidth=0.8
    )
    
    plt.title("Overall Bias Rate Across Vision-Language Architectures", pad=20, weight='bold')
    plt.ylabel("Mean Bias Rate (Prior Influence)", labelpad=10)
    plt.xlabel("")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=30, ha='right')
    
    # Add percentage labels with professional formatting
    for c in ax.containers:
        labels = [f'{w*100:.1f}%' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge', padding=8, weight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig("Figures for Paper/Figure1_Overall_Bias.pdf", bbox_inches='tight')
    plt.close()

def generate_figure2_background(df):
    plt.figure(figsize=(10, 6.5))
    # Standardize order for background to ensure Plain vs Natural comparison is clear
    df['background'] = df['background'].str.capitalize()
    
    ax = sns.barplot(
        data=df, 
        x='model', 
        y='bias_rate', 
        hue='background', 
        capsize=.05, 
        palette={'Plain': '#95A5A6', 'Natural': '#2E86C1'},
        edgecolor='black',
        linewidth=0.8
    )
    
    plt.title("Effect of Environmental Context on Perceptual Bias", pad=20, weight='bold')
    plt.ylabel("Observed Bias Rate", labelpad=10)
    plt.xlabel("")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=30, ha='right')
    
    plt.legend(title="Background Context", loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    
    plt.tight_layout()
    plt.savefig("Figures for Paper/Figure2_Background_Effect.pdf", bbox_inches='tight')
    plt.close()

def generate_figure3_error_dist(df):
    errors = df[df['predicted'] != df['true_legs']].copy()
    
    def cat_error(row):
        try:
            pred = float(row['predicted'])
            if pred == 4: return 'Prior-aligned (pred=4)'
            elif pred in [3, 5]: return 'Off-by-one (pred=3|5)'
            else: return 'Other/Random'
        except:
            return 'Other/Random'
        
    errors['err_type'] = errors.apply(cat_error, axis=1)
    
    dist = errors.groupby(['model', 'err_type']).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
    
    order = ['Prior-aligned (pred=4)', 'Off-by-one (pred=3|5)', 'Other/Random']
    for o in order:
        if o not in dist_pct.columns:
            dist_pct[o] = 0.0
    dist_pct = dist_pct[order]
    
    # Sort rows by Prior-aligned percentage for better visualization
    dist_pct = dist_pct.sort_values(by='Prior-aligned (pred=4)', ascending=False)
    
    COLORS = ['#C0392B', '#F1C40F', '#7F8C8D'] # Stronger, more professional colors
    ax = dist_pct.plot(
        kind='bar', 
        stacked=True, 
        figsize=(10, 6.5), 
        color=COLORS,
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title("Decomposition of Visual Reasoning Errors", pad=20, weight='bold')
    plt.ylabel("Proportion of Total Errors (%)", labelpad=10)
    plt.xlabel("")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title="Error Category", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    
    # Add percentage labels inside bars if they are large enough
    for c in ax.containers:
        labels = [f'{w:.0f}%' if w > 8 else '' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig("Figures for Paper/Figure3_Error_Distribution.pdf", bbox_inches='tight')
    plt.close()

def compute_statistics(df):
    # Load Prior Strengths and drop duplicate column
    priors = pd.read_csv("data/processed/prior_strength_scores.csv").drop(columns=['bias_rate'], errors='ignore')
    
    print("=== Model Architecture vs Prior Override (Spearman ρ) ===")
    for model in df['model'].unique():
        sub = df[df['model'] == model]
        # Aggregate bias per concept
        b_rate = sub.groupby('concept')['bias_rate'].mean().reset_index()
        merged = b_rate.merge(priors, on='concept')
        
        rho, p = spearmanr(merged['prior_strength'], merged['bias_rate'])
        print(f"{model:<25}: ρ = {rho:.3f} | p = {p:.4f}")
    
def generate_figure4_salience(df):
    # Load Prior Strengths
    priors = pd.read_csv("data/processed/prior_strength_scores.csv")
    priors["concept"] = priors["concept"].str.lower()
    
    # Aggregate bias per model and concept
    agg = df.groupby(['model', 'concept'])['bias_rate'].mean().reset_index()
    merged = agg.merge(priors, on='concept')
    
    models = sorted(merged['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Custom annotations for specific salient concepts
    salient_concepts = ['elephant', 'cat', 'binturong', 'numbat', 'fossa']
    
    for idx, mod in enumerate(models):
        ax = axes[idx]
        sub = merged[merged['model'] == mod]
        
        # Plot with regression
        sns.regplot(data=sub, x='prior_strength', y='bias_rate', ax=ax, 
                    scatter_kws={'s': 60, 'alpha': 0.6, 'color': '#2c3e50'},
                    line_kws={'color': '#e74c3c', 'lw': 2})
        
        # Label salient concepts
        for _, row in sub.iterrows():
            if row['concept'] in salient_concepts:
                ax.text(row['prior_strength']+0.01, row['bias_rate'], 
                        row['concept'].capitalize(), fontsize=9, alpha=0.9)
        
        # Stats
        rho, p = spearmanr(sub['prior_strength'], sub['bias_rate'])
        ax.text(0.05, 0.95, f"ρ = {rho:.2f} (p={p:.2f})", transform=ax.transAxes,
                verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})
        
        ax.set_title(mod, fontsize=14, pad=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
        
    fig.text(0.5, 0.02, 'Visual Feature Salience (Prior Strength Score)', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'Observed Bias Rate', va='center', rotation='vertical', fontsize=16)
    
    plt.suptitle("Conceptual Salience vs. Model Prior Override Rate", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    plt.savefig("Figures for Paper/Figure4_Conceptual_Salience.pdf", dpi=300)
    plt.savefig("Figures for Paper/Figure4_Conceptual_Salience.png", dpi=300)
    plt.close()

def run():
    print("Aggregating Unified Multi-Model Dataset...")
    df = load_unified_data()
    
    # Use binary is_biased for standard paper metrics
    df['bias_rate'] = df['is_biased']
    
    print("Computing Table Statistics...")
    compute_statistics(df)
    
    print("Generating Figure 1 (Overall)...")
    generate_figure1_overall(df)
    
    print("Generating Figure 2 (Background)...")
    generate_figure2_background(df)
    
    print("Generating Figure 3 (Errors)...")
    generate_figure3_error_dist(df)
    
    print("Generating Figure 4 (Salience/Conceptual Alignment)...")
    generate_figure4_salience(df)
    
    print("\nAnalysis Complete! Professional PDFs updated in 'Figures for Paper'.")

if __name__ == "__main__":
    run()
