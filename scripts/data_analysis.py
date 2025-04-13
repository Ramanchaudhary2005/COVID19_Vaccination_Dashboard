import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set(style="whitegrid")
plt.style.use("seaborn-v0_8")


data_path = "data/processed/covid_data_cleaned.csv"
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])


os.makedirs("assets/images/eda", exist_ok=True)

# 1. Basic info
print("[INFO] Shape:", df.shape)
print("[INFO] Date Range:", df['date'].min(), "to", df['date'].max())
print("[INFO] Unique Outcomes:", df['outcome'].unique())
print("[INFO] Unique Age Groups:", df['age_group'].unique())
print("[INFO] Columns:\n", df.columns)

# 2. Outcomes Over Time
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='date', y='outcome_unvaccinated', label='Unvaccinated')
sns.lineplot(data=df, x='date', y='outcome_vaccinated', label='Vaccinated')
sns.lineplot(data=df, x='date', y='outcome_boosted', label='Boosted')
plt.title("COVID Outcomes Over Time by Vaccination Status")
plt.ylabel("Outcomes")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
save_path = "assets/images/eda/outcomes_over_time.png"
plt.savefig(save_path)
print(f"[INFO] Saved: {save_path}")
plt.show()


# 3. Total Outcomes by Age Group
age_summary = df.groupby('age_group')[['outcome_unvaccinated', 'outcome_vaccinated', 'outcome_boosted']].sum()
age_summary.plot(kind='bar', figsize=(12,6), stacked=True)
plt.title("Total Outcomes by Age Group and Vaccination Status")
plt.ylabel("Total Outcomes")
plt.xlabel("Age Group")
plt.xticks(rotation=45)
plt.tight_layout()
save_path = "assets/images/eda/age_group_outcomes.png"
plt.savefig(save_path)
print(f"[INFO] Saved: {save_path}")
plt.show()


# 4. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
save_path = "assets/images/eda/correlation_heatmap.png"
plt.savefig(save_path)
print(f"[INFO] Saved: {save_path}")
plt.show()


# 5. Scatter Plot: Vaccinated Ratio vs. Outcomes
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df,
    x='crude_vaccinated_ratio',
    y='outcome_vaccinated',
    hue='age_group',
    palette='Set2',
    alpha=0.7
)
plt.title("Crude Vaccinated Ratio vs. Vaccinated Outcomes")
plt.xlabel("Crude Vaccinated Ratio")
plt.ylabel("Vaccinated Outcomes")
plt.legend(title="Age Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
save_path = "assets/images/eda/scatter_vax_ratio_vs_outcomes.png"
plt.savefig(save_path)
print(f"[INFO] Saved: {save_path}")
plt.show()

# 6. Pair Plot of Selected Features
pairplot_cols = [
    "crude_vaccinated_ratio",
    "crude_boosted_ratio",
    "outcome_unvaccinated",
    "outcome_vaccinated",
    "outcome_boosted"
]

# Reduce size by sampling
df_sampled = df[pairplot_cols].dropna().sample(n=500, random_state=42)

sns.pairplot(df_sampled, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'})
plt.suptitle("Pair Plot of Vaccination Ratios vs. Outcomes", y=1.02)
save_path = "assets/images/eda/pairplot_vax_outcomes.png"
plt.savefig(save_path, bbox_inches='tight')
print(f"[INFO] Saved: {save_path}")
plt.show()

# 7. Box Plot: Outcome Distribution by Age Group (Vaccinated)
plt.figure(figsize=(14, 6))
sns.boxplot(
    data=df,
    x="age_group",
    y="outcome_vaccinated",
    palette="Set3"
)
plt.title("Distribution of Vaccinated Outcomes by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Vaccinated Outcomes")
plt.xticks(rotation=45)
plt.tight_layout()
save_path = "assets/images/eda/boxplot_vaccinated_outcomes.png"
plt.savefig(save_path)
print(f"[INFO] Saved: {save_path}")
plt.show()



