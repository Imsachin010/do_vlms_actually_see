import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import textwrap

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

out_dir = "Figures for Paper"
os.makedirs(out_dir, exist_ok=True)

# 1. Visual Distinctiveness Dictionary
visual_salience = {
    'elephant': 1.0, 'pangolin': 0.9, 'deer': 0.8, 'okapi': 0.7,
    'tiger': 0.7, 'leopard': 0.7, 'cheetah': 0.7, 'cat': 0.7,
    'lion': 0.6, 'horse': 0.6, 'fox': 0.6, 'raccoon': 0.6,
    'hyena': 0.5, 'serval': 0.5, 'dog': 0.4, 'quokka': 0.4,
    'binturong': 0.3, 'numbat': 0.3, 'fossa': 0.2
}

def get_data():
    df = pd.read_csv('results/tables/vlm_test_results.csv')
    return df

def generate_figure_1():
    df = get_data()
    divergence = df.groupby('concept').apply(lambda x: 
        x[x['model'] == 'LLaVA-1.5']['bias_rate'].mean() - 
        x[x['model'] == 'Llama-4-Scout']['bias_rate'].mean()
    ).reset_index()
    divergence.columns = ['concept', 'bias_reduction']
    divergence['salience'] = divergence['concept'].map(visual_salience)
    divergence = divergence.dropna()
    
    plt.figure(figsize=(7, 5))
    ax = sns.regplot(data=divergence, x='salience', y='bias_reduction', 
                     scatter_kws={'s': 50, 'alpha': 0.7},
                     line_kws={'color': 'red'})
    
    # Annotate a few points
    for i, row in divergence.iterrows():
        if row['concept'] in ['elephant', 'cat', 'binturong', 'deer', 'numbat']:
            plt.text(row['salience']+0.02, row['bias_reduction'], row['concept'].capitalize(), fontsize=9)

    plt.title("Bias Reduction by Visual Salience", pad=15)
    plt.xlabel("Visual Feature Salience Score")
    plt.ylabel("Bias Reduction (LLaVA - Llama4)")
    
    rho, p = spearmanr(divergence['salience'], divergence['bias_reduction'])
    plt.text(0.05, 0.95, f"ρ = {rho:.3f}\np = {p:.3f}", transform=ax.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Figure1_Bias_Reduction.pdf"), dpi=300, format='pdf')
    plt.close()

def generate_figure_2():
    df = get_data()
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x='model', y='bias_rate', hue='background', capsize=.1)
    
    plt.title("Background Effect (Legacy vs Modern)")
    plt.ylabel("Bias Rate")
    plt.xlabel("")
    plt.legend(title="Background Context")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Figure2_Background_Effect.pdf"), dpi=300, format='pdf')
    plt.close()

def generate_figure_3():
    llama = pd.read_csv("data/processed/phase2_vlm_results.csv")
    llava = pd.read_csv("data/processed/llava_results.csv")
    llama["model"] = "Llama-4-Scout"
    llava["model"] = "LLaVA-1.5"
    raw = pd.concat([llama, llava])
    
    errors = raw[raw['predicted'] != raw['true_legs']].copy()
    
    def cat_error(row):
        if row['predicted'] == 4: return 'Prior-aligned (pred=4)'
        elif row['predicted'] in [3, 5]: return 'Off-by-one (pred=3|5)'
        else: return 'Other/Random'
        
    errors['err_type'] = errors.apply(cat_error, axis=1)
    
    dist = errors.groupby(['model', 'err_type']).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
    
    order = ['Prior-aligned (pred=4)', 'Off-by-one (pred=3|5)', 'Other/Random']
    for o in order:
        if o not in dist_pct.columns:
            dist_pct[o] = 0.0
    dist_pct = dist_pct[order]
    
    ax = dist_pct.plot(kind='bar', stacked=True, figsize=(7, 5), color=['#d62728', '#f5deb3', '#7f7f7f'])
    
    plt.title("Error Pattern Distribution")
    plt.ylabel("Percentage of Total Errors")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1))
    
    for c in ax.containers:
        labels = [f'{w:.1f}%' if w > 0 else '' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Figure3_Error_Distribution.pdf"), dpi=300, format='pdf')
    plt.close()

def generate_figure_4():
    # 2x4 Qualitative Examples Grid - showcasing unexpected & empirical outputs
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    from PIL import Image
    
    images = [
        # ROW 1: LLaVA-1.5 (Unexpected anomalies & expected overrides)
        ("data/raw/counterfactual_images/leopard_5legs_natural.jpg", "LLaVA-1.5: Random Anomaly\nTrue=5, Pred=2", 5, 2, 'orange'),
        ("data/raw/counterfactual_images/numbat_5legs_plain.png", "LLaVA-1.5: Random Anomaly\nTrue=5, Pred=1", 5, 1, 'orange'),
        ("data/raw/counterfactual_images/elephant_5legs_natural.png", "LLaVA-1.5: Expected Bias\nTrue=5, Pred=4", 5, 4, 'red'),
        ("data/raw/counterfactual_images/cat_3legs_plain.png", "LLaVA-1.5: Expected Bias\nTrue=3, Pred=4", 3, 4, 'red'),
        
        # ROW 2: Llama-4-Scout (Unexpected failures & successful overrides)
        ("data/raw/counterfactual_images/cheetah_5legs_plain.png", "Llama-4: Severe Miscount\nTrue=5, Pred=8", 5, 8, 'purple'),
        ("data/raw/counterfactual_images/cat_3legs_natural.png", "Llama-4: Unexpected Prior Collapse\nTrue=3, Pred=4", 3, 4, 'red'),
        ("data/raw/counterfactual_images/cat_5legs_plain.png", "Llama-4: Off-by-one Count\nTrue=5, Pred=3", 5, 3, 'orange'),
        ("data/raw/counterfactual_images/deer_3legs_plain.png", "Llama-4: Successful Count\nTrue=3, Pred=3", 3, 3, 'green')
    ]
    
    for idx, ax in enumerate(axes.flat):
        path, title, true_legs, pred, color = images[idx]
        try:
            img = Image.open(path)
            ax.imshow(img)
            ax.axis('off')
            
            ax.set_title(title, fontsize=11, color='black', pad=10, backgroundcolor='#f0f0f0')
            ax.text(0.05, 0.05, f"pred={pred} (true={true_legs})", color=color, 
                    fontsize=14, weight='bold', transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
        except Exception as e:
            ax.text(0.5, 0.5, f"Image Missing", ha='center', va='center')
            ax.axis('off')

    plt.suptitle("Qualitative Override Gallery (Unexpected Result Breakdown)", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(out_dir, "Figure4_Qualitative_Examples.pdf"), dpi=300, format='pdf')
    plt.close()

def run():
    print("Generating Figure 1 (PDF)...")
    generate_figure_1()
    print("Generating Figure 2 (PDF)...")
    generate_figure_2()
    print("Generating Figure 3 (PDF)...")
    generate_figure_3()
    print("Generating Figure 4 (PDF)...")
    generate_figure_4()
    print("All figures successfully saved to 'Figures for Paper' in PDF format!")

if __name__ == "__main__":
    run()
