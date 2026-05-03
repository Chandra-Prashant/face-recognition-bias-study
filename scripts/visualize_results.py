import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# 1. ROBUST PATH LOGIC
# This ensures the script works regardless of whether you run it from /scripts or the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == 'scripts':
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "bias_metrics_optimized.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🔍 Checking path: {RESULTS_FILE}")

# 2. LOAD DATA
try:
    df = pd.read_csv(RESULTS_FILE)
    print(f"✅ Successfully loaded {len(df)} records for visualization.")
except FileNotFoundError:
    print(f"❌ ERROR: File not found at {RESULTS_FILE}")
    print("Please ensure evaluate_bias.py has finished running successfully.")
    exit()

# 3. ENHANCED VISUALIZATION
plt.figure(figsize=(16, 9))
sns.set_style("whitegrid")

# Professional Architectural Paradigm colors
paper_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Use 'group' (Race_Gender) for intersectional depth
ax = sns.boxplot(
    x='group', 
    y='similarity', 
    hue='model', 
    data=df,
    palette=paper_colors,
    linewidth=1.5,
    fliersize=2
)

# 4. OPERATIONAL THRESHOLD (The 24.9% Failure Benchmark)
# This explains the failure rate mentioned in your results
plt.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Security Threshold (T=0.85)')

# Professional Formatting
plt.title('Intersectional Robustness Analysis Across Three Architectural Paradigms', fontsize=18, pad=25)
plt.ylabel('Cosine Similarity (Operational Score)', fontsize=14)
plt.xlabel('Intersectional Demographic Group (Race_Gender)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Architecture', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

# Add Annotation for the "Vulnerability Zone"
plt.text(x=0.5, y=0.3, s="Failure Zone (Similarity < 0.85)", color='red', 
         fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())

plt.tight_layout()

# Save for 6-page draft inclusion
output_path = os.path.join(OUTPUT_DIR, "intersectional_robustness_boxplot.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ High-resolution boxplot saved to: {output_path}")
plt.show()

# 5. STATISTICAL SIGNIFICANCE (ANOVA Verification)
print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE TEST: IMPACT OF DEMOGRAPHICS")
print("="*60)

for model_name in df['model'].unique():
    model_df = df[df['model'] == model_name]
    # Grouping by 'group' to test intersectional variance
    groups = [data['similarity'].values for name, data in model_df.groupby('group')]
    
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        status = "SIGNIFICANT BIAS" if p_val < 0.05 else "INSIGNIFICANT"
        print(f"[+] Architecture: {model_name:10} | p-value: {p_val:.4e} | {status}")