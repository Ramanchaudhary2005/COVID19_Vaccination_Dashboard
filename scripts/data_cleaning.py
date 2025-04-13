import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_missing_values(df, save_path="assets/images/missing_values.png"):
    plt.figure(figsize=(12, 6))
    missing = df.isnull().sum().sort_values(ascending=False)
    sns.barplot(x=missing.values, y=missing.index, palette="viridis")
    plt.title("Missing Values per Column")
    plt.xlabel("Count")
    plt.ylabel("Columns")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print("[INFO] Missing value plot saved to:", save_path)


def clean_covid_data(input_path, output_path):
    
    df = pd.read_csv(input_path)

    
    if 'Week End' in df.columns:
        df.rename(columns={'Week End': 'date'}, inplace=True)

    print("[INFO] Original shape:", df.shape)

    
    visualize_missing_values(df)

    # Convert 'date' col
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Clean col
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_").replace("-", "_"), inplace=True)

    # Drop rows 
    df.dropna(subset=['date'], inplace=True)

    # Fill num
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Save hist before fill
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"{col} - Before Filling NAs")
            img_path = f"assets/images/before_fill_{col}.png"
            plt.savefig(img_path)
            plt.close()
            print(f"[INFO] Saved histogram before fill for: {col} -> {img_path}")

    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Save cln ver
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("[INFO] Cleaned data saved to:", output_path)
    return df

if __name__ == "__main__":
    input_file = "data/raw/COVID-19_Outcomes_by_Vaccination_Status_-_Historical.csv"
    output_file = "data/processed/covid_data_cleaned.csv"
    clean_covid_data(input_file, output_file)



