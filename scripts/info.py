import pandas as pd


df = pd.read_csv("data/processed/covid_data_cleaned.csv")  


print("Dataset Info:")
df.info()



print("\nMissing Values in Each Column:")
print(df.isnull().sum())


print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())
print(df.describe())
print(df.head())
