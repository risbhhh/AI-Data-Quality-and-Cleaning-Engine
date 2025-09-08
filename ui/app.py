import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
from ingest.loaders import load_file
from profiling.profiler import clean_dataset   # <-- import here

st.title("AI Data Quality & Cleaning Engine")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_file(uploaded_file)
    st.write("### Raw Data Preview", df.head())

    if st.button("Run Cleaning Pipeline"):
        cleaned = clean_dataset(df)
        st.write("### Cleaned Data Preview", cleaned.head())

        # Export
        csv = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned CSV",
            csv,
            "cleaned_dataset.csv",
            "text/csv",
            key="download-csv"
        )
