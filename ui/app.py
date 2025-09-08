import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # make repo root importable

import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

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
    # Profiling
    # -----------------------------
    st.write("### 📊 Data Profiling Report")
    try:
        profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
        st_profile_report(profile)
    except Exception as e:
        st.error(f"Profiling failed: {e}")

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
