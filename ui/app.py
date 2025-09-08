import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # make repo root importable

import streamlit as st
import pandas as pd

from ingest.loaders import load_file
from profiling.profiler import clean_dataset

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Data Quality & Cleaning Engine", layout="wide")
st.title("🧹 AI Data Quality & Cleaning Engine")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = load_file(uploaded_file)
    st.write("### 🔎 Raw Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Custom Lightweight Profiling
    # -----------------------------
    st.write("### 📊 Data Quality Report")

    # Missing values
    st.write("**Missing Values (%):**")
    st.dataframe(df.isnull().mean() * 100)

    # Basic stats
    st.write("**Numeric Summary:**")
    st.dataframe(df.describe())

    # Categorical counts
    st.write("**Categorical Columns (Top 10 values):**")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        st.write(f"🔹 {col}")
        st.write(df[col].value_counts().head(10))

    # -----------------------------
    # Cleaning Pipeline
    # -----------------------------
    if st.button("✨ Run Cleaning Pipeline"):
        cleaned = clean_dataset(df)
        st.write("### ✅ Cleaned Data Preview")
        st.dataframe(cleaned.head())

        # Download button
        csv = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Cleaned CSV",
            csv,
            "cleaned_dataset.csv",
            "text/csv",
            key="download-csv"
        )
