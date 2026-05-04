# Credit Risk Model Validation Suite — Streamlit App

Interactive dashboard for credit risk model validation.

## Files

- `app_en.py` — English version (primary, for US/LATAM market)
- `app_es.py` — Spanish version (for Argentina/LATAM market)
- `requirements.txt` — dependencies for Streamlit Cloud

## Deploy to Streamlit Cloud (free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set **Main file path** to `app_en.py` (or `app_es.py`)
5. Click Deploy

The app runs in **demo mode** automatically if model artifacts
are not present — no data upload needed for Streamlit Cloud.

## Run locally with real model

```bash
# From the project root (homecredit-model-validation/)
pip install -r streamlit_app/requirements.txt

# English version
streamlit run streamlit_app/app_en.py

# Spanish version
streamlit run streamlit_app/app_es.py
```

## Demo mode vs Live mode

| Mode | Trigger | Data |
|------|---------|------|
| Demo | `models/champion/model.pkl` not found | Synthetic (realistic) |
| Live | Model artifacts present | Real Home Credit predictions |

In demo mode all visualizations work with synthetic data that
mirrors the real dataset's statistical properties.
The Threshold Optimizer, SR 11-7 validation, Fairness analysis
and Drift monitor all function fully in demo mode.

## Sections

| Section | What it shows |
|---------|--------------|
| Overview | KPIs, ROC curve, stress test results |
| SR 11-7 Validation | Gini, KS, calibration, PSI, sensitivity |
| Fairness Analysis | DPD, DIR, Equalized Odds by gender and age |
| Threshold Optimizer | Optimal threshold by market with live parameters |
| Loan Simulator | Individual prediction with SHAP (live mode only) |
| Drift Monitor | PSI trend and retirement criteria |
| Model Card | Full regulatory documentation |
