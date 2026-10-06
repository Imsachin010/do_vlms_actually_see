import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
out_dir = "Figures for Paper"
os.makedirs(out_dir, exist_ok=True)

def run():
    df = pd.read_csv('results/tables/vlm_test_results.csv')
    
    # Calculate overall means
    overall = df.groupby('model')['bias_rate'].mean().reset_index()
    
    plt.figure(figsize=(5, 5))
    ax = sns.barplot(data=overall, x='model', y='bias_rate', palette=['#1f77b4', '#ff7f0e'])
    
    plt.title("Overall Bias Rate Comparison", pad=15)
    plt.ylabel("Overall Bias Rate")
    plt.xlabel("")
    plt.ylim(0, 1)
    
    # Annotate percentages
    for c in ax.containers:
        labels = [f'{w*100:.1f}%' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge', padding=5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Figure1_Overall_Bias.pdf"), dpi=300, format='pdf')
    plt.close()

if __name__ == "__main__":
    run()
