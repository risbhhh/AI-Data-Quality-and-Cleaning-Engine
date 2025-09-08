import pandas as pd
from typing import Union

def load_file(file: Union[str, 'io.BytesIO']):
    """Load CSV or Parquet into a pandas DataFrame.

    Accepts a path string or an uploaded file-like object from Streamlit.
    """
    if isinstance(file, str):
        if file.endswith('.csv'):
            return pd.read_csv(file)
        if file.endswith('.parquet'):
            return pd.read_parquet(file)
    # assume file-like (Streamlit)
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_parquet(file)
