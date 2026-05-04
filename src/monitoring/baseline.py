"""
Drift Monitoring — Home Credit Default Risk
===========================================
Establece baseline de distribución de scores.
En producción: correr semanalmente para detectar drift.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

MODELS_DIR  = Path("models/champion")
DATA_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports")


def compute_psi(expected, actual, n_bins=10) -> float:
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6; bins[-1] += 1e-6
    def bucket(d):
        c, _ = np.histogram(d, bins=bins)
        p = c / len(d)
        return np.where(p == 0, 1e-4, p)
    e, a = bucket(expected), bucket(actual)
    return round(float(np.sum((a - e) * np.log(a / e))), 4)


def create_baseline():
    logger.info("=" * 50)
    logger.info("STAGE 7/7 — Drift Baseline")
    logger.info("=" * 50)

    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").iloc[:, 0]

    ref_scores = model.predict_proba(X_train)[:, 1]

    ref_df = X_train.copy()
    ref_df["predicted_score"] = ref_scores
    ref_df["actual_label"]    = y_train.values
    ref_df.to_parquet(MODELS_DIR / "reference_data.parquet", index=False)

    config = {
        "psi_alert_threshold":   0.10,
        "psi_retrain_threshold": 0.25,
        "reference_score_mean":  float(ref_scores.mean()),
        "reference_score_std":   float(ref_scores.std()),
        "reference_default_rate": float(y_train.mean()),
        "n_reference_rows":      int(len(ref_df)),
    }
    with open(MODELS_DIR / "drift_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.success(
        f"Baseline OK — {len(ref_df):,} filas | "
        f"score medio: {ref_scores.mean():.4f}"
    )


def check_drift(new_data_path: str = None) -> dict:
    """Compara nueva data contra baseline. Retorna reporte de drift."""
    with open(MODELS_DIR / "model.pkl",      "rb") as f: model  = pickle.load(f)
    with open(MODELS_DIR / "drift_config.json")      as f: config = json.load(f)

    reference  = pd.read_parquet(MODELS_DIR / "reference_data.parquet")
    ref_scores = reference["predicted_score"].values

    if new_data_path:
        new_data = pd.read_parquet(new_data_path)
    else:
        # Simulación de drift para demo
        X_test    = pd.read_parquet(DATA_DIR / "X_test.parquet")
        new_data  = X_test.copy()
        num_cols  = new_data.select_dtypes(include=np.number).columns
        new_data[num_cols] += np.random.normal(0, 0.05, new_data[num_cols].shape)

    new_scores  = model.predict_proba(new_data)[:, 1]
    score_psi   = compute_psi(ref_scores, new_scores)

    feature_psi = {}
    for feat in new_data.select_dtypes(include=np.number).columns[:10]:
        if feat in reference.columns:
            try:
                feature_psi[feat] = compute_psi(reference[feat].values, new_data[feat].values)
            except Exception:
                pass

    alert = ("ok" if score_psi <= config["psi_alert_threshold"]
             else "monitor" if score_psi <= config["psi_retrain_threshold"]
             else "retrain_required")

    report = {
        "score_psi":   score_psi,
        "alert_level": alert,
        "thresholds":  {"alert": config["psi_alert_threshold"],
                        "retrain": config["psi_retrain_threshold"]},
        "top_drifting_features": dict(
            sorted(feature_psi.items(), key=lambda x: -x[1])[:5]
        ),
        "recommendation": {
            "ok":               "Sin acción requerida",
            "monitor":          "Aumentar frecuencia de monitoreo — investigar features que derivan",
            "retrain_required": "Reentrenar el modelo inmediatamente",
        }[alert],
    }

    with open(REPORTS_DIR / "drift_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Drift check: PSI={score_psi:.4f} | Status: {alert.upper()}")
    return report


if __name__ == "__main__":
    create_baseline()
