# Home Credit — Model Validation Suite

Pipeline end-to-end de validación de modelos crediticios bajo **SR 11-7**,
con **MLOps**, **explainability (SHAP)**, **fairness** y **AI Governance**.

## Dataset

**Home Credit Default Risk** — Kaggle  
URL: https://www.kaggle.com/c/home-credit-default-risk/data  
Archivo requerido: `application_train.csv` → colocar en `data/raw/`

## Quickstart

```bash
# 1. Clonar y crear entorno
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar el CSV
cp ~/Downloads/application_train.csv data/raw/

# 4. Correr pipeline completo
python run_all.py

# 5. Abrir notebooks en orden
jupyter notebook notebooks/01_eda.ipynb

# 6. Lanzar API (opcional)
uvicorn src.api.main:app --reload --port 8000
```

## Estructura

```
homecredit-model-validation/
├── src/
│   ├── data/ingest.py          # Stage 1: ingesta + DQ checks
│   ├── features/build.py       # Stage 2: feature engineering
│   ├── models/train.py         # Stage 3: XGBoost + baseline LR
│   ├── validation/
│   │   ├── sr117.py            # Stage 4: validación SR 11-7
│   │   └── explain.py          # Stage 6: SHAP
│   ├── governance/
│   │   ├── fairness.py         # Stage 5: fairness + bias
│   │   └── model_card.py       # Model Card auto-generado
│   ├── monitoring/baseline.py  # Stage 7: drift baseline
│   └── api/main.py             # FastAPI con SHAP por predicción
├── notebooks/
│   ├── 01_eda.ipynb            # Análisis exploratorio
│   ├── 02_threshold_analysis.ipynb
│   ├── 03_error_analysis.ipynb
│   ├── 04_robustez_counterfactuals.ipynb
│   └── 05_lifecycle_simulation.ipynb
├── docs/decisions/ADRs.py      # Architectural Decision Records
├── data/raw/                   # CSVs de Home Credit (no versionados)
├── models/champion/            # Artefactos del modelo
├── reports/                    # Reportes y figuras auto-generados
├── run_all.py                  # Entrada principal del pipeline
└── requirements.txt
```

## Outputs del Pipeline

| Archivo | Contenido |
|---------|-----------|
| `models/champion/model.pkl` | Modelo champion (XGBoost) |
| `models/champion/shap_explainer.pkl` | SHAP TreeExplainer |
| `reports/validation_metrics.json` | Gini, AUC, KS, PSI |
| `reports/sr117_validation.json` | Reporte SR 11-7 completo |
| `reports/fairness_report.json` | DPD, DIR, Equalized Odds |
| `reports/model_cards/model_card.html` | Model Card |
| `reports/figures/` | ROC, calibración, SHAP, fairness |

## Notebooks — Orden de trabajo

Correr `python run_all.py` primero. Después abrir las notebooks en orden:

1. **01_eda.ipynb** — EDA completo. Completar celdas ✏️ con tus conclusiones.
2. **02_threshold_analysis.ipynb** — Costo-beneficio del threshold.
3. **03_error_analysis.ipynb** — Dónde y por qué falla el modelo.
4. **04_robustez_counterfactuals.ipynb** — Consistencia y counterfactuals.
5. **05_lifecycle_simulation.ipynb** — Simulación de 6 meses en producción.

## Stack

- **ML**: XGBoost, scikit-learn
- **MLOps**: MLflow, DVC-ready
- **Explainability**: SHAP
- **Fairness**: métricas propias (DPD, DIR, Equalized Odds)
- **API**: FastAPI + Pydantic
- **Regulatorio**: SR 11-7, EU AI Act, BCRA Com. A 7724
