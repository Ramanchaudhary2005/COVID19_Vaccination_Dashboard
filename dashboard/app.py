import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path


st.set_page_config(page_title="COVID-19 Vaccination Dashboard", layout="wide")
st.title("💉 COVID-19 Vaccination Dashboard")
st.markdown("Analyzing vaccination outcomes across age groups over time.")

# L_D
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "covid_data_cleaned.csv"
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

#side bar 
st.sidebar.header("📊 Filters")
age_group = st.sidebar.multiselect("Select Age Group(s):", df["age_group"].unique(), default=df["age_group"].unique())
outcome_type = st.sidebar.selectbox("Select Outcome Type:", ["outcome_unvaccinated", "outcome_vaccinated", "outcome_boosted"])

# Fil data
filtered_df = df[df["age_group"].isin(age_group)]

# Line Chart
st.subheader("📈 Outcomes Over Time")
fig1, ax1 = plt.subplots(figsize=(12, 5))
sns.lineplot(data=filtered_df, x="date", y=outcome_type, hue="age_group", ax=ax1)
ax1.set_title(f"{outcome_type.replace('_', ' ').title()} Over Time")
ax1.set_ylabel("Count")
st.pyplot(fig1)


st.subheader("🖼️ Exploratory Data Analysis (EDA) Plots")
eda_folder = Path(__file__).resolve().parents[1] / "assets" / "images" / "eda"

if eda_folder.exists():
    plots = [f for f in os.listdir(eda_folder) if f.endswith(".png")]
    for plot in plots:
        st.image(os.path.join(eda_folder, plot), use_column_width=True, caption=plot.replace("_", " ").replace(".png", "").title())
else:
    st.warning("No EDA plots found. Please generate them using the analysis script.")

