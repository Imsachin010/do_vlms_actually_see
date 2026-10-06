import pandas as pd
from scipy import stats

def run():
    df = pd.read_csv('results/tables/vlm_test_results.csv')
    
    print("\n=== Tier 1 vs Tier 3 Analysis (Independent t-test) ===")
    for model in df['model'].unique():
        for bg in ['plain', 'natural', 'all']:
            if bg == 'all':
                sub = df[df['model'] == model]
            else:
                sub = df[(df['model'] == model) & (df['background'] == bg)]
                
            tier1 = sub[sub['tier'] == 'High']['bias_rate']
            tier3 = sub[sub['tier'] == 'Low']['bias_rate']
            
            t, p = stats.ttest_ind(tier1, tier3, nan_policy='omit')
            
            # Print everything regardless of significance to see the trend
            sig = '***' if p < 0.05 else ('*' if p < 0.1 else '')
            print(f"[{model:15s} | {bg:7s}] Tier1={tier1.mean():.2f} vs Tier3={tier3.mean():.2f} | p={p:.4f} {sig}")


    print("\n=== Top 5 vs Bottom 5 Concepts Analysis ===")
    # Get distinct concepts sorted by prior strength
    concepts = df[['concept', 'prior_strength']].drop_duplicates().sort_values('prior_strength', ascending=False)
    top_5 = concepts.head(5)['concept'].tolist()
    bot_5 = concepts.tail(5)['concept'].tolist()
    
    print(f"Top 5 Prior: {', '.join(top_5)}")
    print(f"Bot 5 Prior: {', '.join(bot_5)}")
    print()
    
    for model in df['model'].unique():
        for bg in ['plain', 'natural', 'all']:
            if bg == 'all':
                sub = df[df['model'] == model]
            else:
                sub = df[(df['model'] == model) & (df['background'] == bg)]
                
            top_bias = sub[sub['concept'].isin(top_5)]['bias_rate']
            bot_bias = sub[sub['concept'].isin(bot_5)]['bias_rate']
            
            t, p = stats.ttest_ind(top_bias, bot_bias, nan_policy='omit')
            
            sig = '***' if p < 0.05 else ('*' if p < 0.1 else '')
            print(f"[{model:15s} | {bg:7s}] Top5={top_bias.mean():.2f} vs Bot5={bot_bias.mean():.2f} | p={p:.4f} {sig}")

if __name__ == "__main__":
    run()
