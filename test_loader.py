from ingest.loaders import load_file
import pandas as pd
import io

def test_csv_load():
    s = 'a,b\n1,2\n3,4'
    f = io.StringIO(s)
    df = load_file(f)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2,2)
