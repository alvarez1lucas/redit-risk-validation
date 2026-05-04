"""
Explainability — SHAP
=====================
Genera explicaciones globales (summary) y locales (por predicción).
Guarda el explainer para uso en la API.
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

MODELS_DIR  = Path("models/champion")
DATA_DIR    = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_explainer(model, X_train):
    try:
        explainer = shap.TreeExplainer(model, X_train)
        logger.info("  TreeExplainer cargado (fast path)")
    except Exception:
        logger.warning("  Fallback a KernelExplainer (más lento)")
        bg = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
    return explainer


def explain():
    logger.info("=" * 50)
    logger.info("STAGE 6/7 — Explainability (SHAP)")
    logger.info("=" * 50)

    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")

    with open(DATA_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    explainer = build_explainer(model, X_train)

    # SHAP sobre muestra del test (500 para velocidad)
    sample_size = min(500, len(X_test))
    X_sample    = X_test.sample(sample_size, random_state=42)
    logger.info(f"  Calculando SHAP para {sample_size} muestras...")

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, color="#534AB7")
    plt.title("Importancia global de features (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm — dirección y magnitud")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Guardar explainer
    with open(MODELS_DIR / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    # Ejemplo de explicación individual
    ev = explainer.expected_value
    if isinstance(ev, list): ev = ev[1]
    sv_row = explainer.shap_values(X_test.iloc[[0]])
    if isinstance(sv_row, list): sv_row = sv_row[1]
    sv_row = sv_row.flatten()
    top5 = sorted(zip(feature_names, sv_row), key=lambda x: -abs(x[1]))[:5]
    sample_exp = {
        "base_value":  round(float(ev), 4),
        "top_factors": [
            {"feature": f.replace("num__","").replace("cat__",""),
             "shap_value": round(float(v), 4),
             "direction": "aumenta_riesgo" if v > 0 else "reduce_riesgo"}
            for f, v in top5
        ]
    }
    with open(MODELS_DIR / "sample_explanation.json", "w") as f:
        json.dump(sample_exp, f, indent=2)

    logger.success("SHAP OK — explainer y figuras guardados")


if __name__ == "__main__":
    explain()
