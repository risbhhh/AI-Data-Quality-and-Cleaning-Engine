import pandas as pd
import numpy as np

def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a profile report: missing values, duplicates, anomalies"""
    report = {}

    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = {col: int(val) for col, val in missing.items() if val > 0}

    # Duplicates
    duplicates = df.duplicated().sum()
    report["duplicates"] = int(duplicates)

    # Anomalies (z-score > 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    anomalies = {}
    for col in numeric_cols:
        if df[col].std() > 0:  # avoid division by zero
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            anomalies[col] = int((abs(z_scores) > 3).sum())
    report["anomalies"] = anomalies

    return report
