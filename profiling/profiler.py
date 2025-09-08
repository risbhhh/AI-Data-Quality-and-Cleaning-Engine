import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    # Fill missing Age with median
    if "Age" in cleaned_df.columns:
        cleaned_df["Age"] = cleaned_df["Age"].fillna(cleaned_df["Age"].median())

    # Fill Embarked missing with mode
    if "Embarked" in cleaned_df.columns and cleaned_df["Embarked"].notna().any():
        cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(cleaned_df["Embarked"].mode()[0])

    # Replace missing Cabin with "Unknown"
    if "Cabin" in cleaned_df.columns:
        cleaned_df["Cabin"] = cleaned_df["Cabin"].fillna("Unknown")

    # Cap Fare at 95th percentile
    if "Fare" in cleaned_df.columns:
        fare_cap = cleaned_df["Fare"].quantile(0.95)
        cleaned_df["Fare"] = cleaned_df["Fare"].clip(upper=fare_cap)

    # Drop messy columns if present
    for col in ["Ticket", "Name"]:
        if col in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=[col])

    # Encode Sex
    if "Sex" in cleaned_df.columns:
        cleaned_df["Sex"] = cleaned_df["Sex"].map({"male": 1, "female": 0})

    # One-hot encode Embarked
    if "Embarked" in df.columns:  # original df, before we dropped
        cleaned_df = pd.get_dummies(cleaned_df, columns=["Embarked"], prefix="Embarked")

    return cleaned_df
