import pandas as pd

def profile_df(df: pd.DataFrame):
    return {
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_fraction': df.isnull().mean().to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'head': df.head(5).to_dict(orient='records')
    }
