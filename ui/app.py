import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from ingest.loaders import load_file
from profiling.profiler import profile_dataset
from cleaning.cleaner import clean_dataset
from llm.script_generator import generate_cleaning_script

st.set_page_config(page_title="AI Data Quality & Cleaning Engine", layout="wide")
st.title("🤖 AI Data Quality & Cleaning Engine")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_file(uploaded_file)
    st.write("### 🔎 Raw Data Preview", df.head())

    # Profiling
    st.write("### 📊 Data Quality Report")
    report = profile_dataset(df)
    st.json(report)

    # LLM cleaning script
    if st.checkbox("💡 Generate Pandas Cleaning Script (LLM)"):
        try:
            code = generate_cleaning_script(report, df.head().to_string())
            st.code(code, language="python")
        except Exception as e:
            st.error(f"LLM script generation failed: {e}")

    # Run cleaning pipeline
    if st.button("✨ Run Auto-Cleaning Engine"):
        cleaned = clean_dataset(df)
        st.write("### ✅ Cleaned Data Preview", cleaned.head())

        csv = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned CSV", csv, "cleaned_dataset.csv", "text/csv")
