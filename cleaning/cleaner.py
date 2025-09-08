import pandas as pd

def clean_dataset(df: pd.DataFrame, drop_duplicates=True) -> pd.DataFrame:
    """Basic cleaning: remove duplicates, fill missing values"""
    cleaned = df.copy()

    # Drop duplicates
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    # Fill numeric NaNs with median
    for col in cleaned.select_dtypes(include=["float64", "int64"]).columns:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Fill categorical NaNs with mode
    for col in cleaned.select_dtypes(include=["object", "category"]).columns:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    return cleaned
