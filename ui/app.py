import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
from ingest.loaders import load_file
from profiling.profiler import profile_df
import numpy as np
from ml.autoencoder import TabularAutoencoder
from ml.inference import detect_anomalies

st.set_page_config(page_title='AI Data QC Engine', layout='wide')

st.title('AI Data Quality & Cleaning Engine — Demo')

uploaded = st.file_uploader('Upload CSV or Parquet', type=['csv','parquet'])
sample_button = st.button('Load sample Titanic CSV')

if sample_button:
    # create a tiny sample DF
    df = pd.DataFrame({
        'age': [22, 38, 26, None, 35],
        'fare': [7.25, 71.2833, 7.925, 8.05, None],
        'sex': ['male','female','female','male','female'],
        'survived':[0,1,1,0,1]
    })
    st.session_state['df'] = df

if uploaded:
    try:
        df = load_file(uploaded)
        st.session_state['df'] = df
    except Exception as e:
        st.error(f'Error loading file: {e}')

if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader('Profiling')
    prof = profile_df(df)
    st.json(prof)

    st.subheader('Data preview')
    st.dataframe(df.head(50))

    if st.button('Run simple ML detection (Autoencoder demo)'):
        numeric = df.select_dtypes(include=['number']).fillna(0)
        if numeric.shape[1] == 0:
            st.warning('No numeric columns to run demo.')
        else:
            X = (numeric - numeric.mean()) / (numeric.std()+1e-8)
            X_np = X.to_numpy()
            n_features = X_np.shape[1]
            model = TabularAutoencoder(n_features=n_features, hidden_dim=max(8, n_features*2))
            # random init -> use high threshold so few flagged; in real use, load trained model
            mask, errors = detect_anomalies(model, X_np, threshold=0.1)
            res = pd.DataFrame({'_row': np.arange(len(errors)), 'recon_err': errors, 'anomaly': mask})
            st.write('Anomaly scores (toy):')
            st.dataframe(res)
            st.download_button('Download anomaly CSV', data=res.to_csv(index=False), file_name='anomalies.csv')

    if st.button('Export cleaned CSV (no-op demo)'):
        st.download_button('Download CSV', data=df.to_csv(index=False), file_name='cleaned.csv')
else:
    st.info('Upload or load a sample CSV to start the demo.')
