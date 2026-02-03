import streamlit as st
import pandas as pd
import os

# App title
st.set_page_config(page_title="Salary Data Analysis", layout="wide")
st.title("📊 Salary Data Analysis App")

# Show files in directory (debug – safe to keep or remove later)
st.write("📁 Files in app directory:")
st.write(os.listdir())

# Load dataset (CSV must be in SAME folder as this file)
try:
    df = pd.read_csv("ML-P4-Salary_Data.csv")
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ CSV file not found. Please check file name and location.")
    st.stop()

# Display data
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Basic info
st.subheader("📈 Dataset Information")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# Show column names
st.subheader("🧾 Column Names")
st.write(list(df.columns))

# Simple analysis (safe for any salary dataset)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

if len(numeric_cols) > 0:
    st.subheader("📊 Summary Statistics")
    st.write(df[numeric_cols].describe())
else:
    st.warning("No numeric columns found for analysis.")


