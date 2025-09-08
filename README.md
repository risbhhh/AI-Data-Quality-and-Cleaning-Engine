# AI Data Quality & Cleaning Engine

Minimal, demo-ready repository combining:
- pandas-based ingest & profiling
- a simple Streamlit UI to upload & profile CSVs
- placeholder PyTorch autoencoder for anomaly detection
- OR-Tools optimizer example
- LangChain integration stub (optional)

## Quickstart

1. Create a virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Run Streamlit demo: `streamlit run ui/app.py`
4. Try uploading a CSV (e.g., Titanic dataset) and click "Run ML Detection".

## Structure

```
ai-data-cleaner/
├─ ingest/
├─ profiling/
├─ rules/
├─ ml/
├─ llm/
├─ optimize/
├─ ui/
├─ notebooks/
├─ tests/
└─ .github/
```

## Notes
- LangChain/LLM calls are optional and disabled by default.
- This repo is a starting point — expand models, unit tests, and CI as needed.
