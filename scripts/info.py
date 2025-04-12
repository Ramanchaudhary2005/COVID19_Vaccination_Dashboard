import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/covid_data_cleaned.csv")  # Adjust path if needed

# Display general information about the dataset
print("Dataset Info:")
df.info()


# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())
print(df.describe())
print(df.head())
