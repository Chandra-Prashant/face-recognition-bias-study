import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "..", "results", "bias_metrics.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "results", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load and Prepare Data
df = pd.read_csv(RESULTS_FILE)
# Create intersectional labels (e.g., "Black Female")
df['intersectional_group'] = df['race'] + "_" + df['gender']

print("📊 Data Overview:")
print(df.groupby(['intersectional_group', 'model'])['similarity'].mean().unstack())

# 2. Statistical Analysis: Two-Way ANOVA
# We test: Does the interaction between Group and Model significantly affect Similarity?
model = ols('similarity ~ C(intersectional_group) + C(model) + C(intersectional_group):C(model)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

anova_path = os.path.join(OUTPUT_DIR, "anova_results.csv")
anova_table.to_csv(anova_path)
print(f"\n✅ ANOVA Results saved to {anova_path}")
print(anova_table)

# 3. Visualization: Intersectional Heatmap
# This shows the "Reliability Gap" across different conditions
plt.figure(figsize=(12, 8))
pivot_df = df.pivot_table(index='intersectional_group', columns='perturbation', values='similarity', aggfunc='mean')
sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.8)

plt.title('Intersectional Robustness Heatmap (Average Similarity)')
plt.ylabel('Demographic Group (Race + Gender)')
plt.xlabel('Environmental Degradation Type')
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, "intersectional_heatmap.png")
plt.savefig(heatmap_path)
print(f"✅ Heatmap saved to {heatmap_path}")

# 4. Summary Table for Report
summary = df.groupby(['race', 'perturbation'])['similarity'].mean().unstack()
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_by_race.csv"))