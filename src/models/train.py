"""
Training pipeline — Home Credit Default Risk
=============================================
Entrena XGBoost (champion) vs Logistic Regression (baseline SR 11-7).
Registra experimentos en MLflow.
Selecciona champion por Gini sobre test set.
"""
import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_PROCESSED = Path("data/processed")
MODELS_DIR     = Path("models/champion")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)


def gini(y_true, y_score) -> float:
    return round(2 * roc_auc_score(y_true, y_score) - 1, 4)


def evaluate(model, X, y, name: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.50).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return {
        "model":         name,
        "gini":          gini(y, proba),
        "auc_roc":       round(float(roc_auc_score(y, proba)), 4),
        "avg_precision": round(float(average_precision_score(y, proba)), 4),
        "precision":     round(tp / (tp + fp + 1e-9), 4),
        "recall":        round(tp / (tp + fn + 1e-9), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def train_baseline(X_train, y_train):
    """
    Logistic Regression — baseline obligatorio por SR 11-7.
    El modelo complejo debe demostrar mejora sobre este benchmark.
    """
    model = LogisticRegression(
        C=0.1, max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="saga",
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    XGBoost con early stopping.
    scale_pos_weight maneja el desbalance de clases de Home Credit (~8% default).
    """
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(f"  scale_pos_weight = {ratio:.2f} (ratio no-default/default)")

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        scale_pos_weight=ratio,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    logger.info(f"  Best iteration: {model.best_iteration}")
    return model


def train():
    logger.info("=" * 50)
    logger.info("STAGE 3/7 — Training")
    logger.info("=" * 50)

    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet").iloc[:, 0]
    y_test  = pd.read_parquet(DATA_PROCESSED / "y_test.parquet").iloc[:, 0]

    mlflow.set_experiment("homecredit-model-validation")
    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="logistic_baseline"):
        logger.info("Entrenando baseline (Logistic Regression)...")
        lr      = train_baseline(X_train, y_train)
        lr_m    = evaluate(lr, X_test, y_test, "logistic_baseline")
        mlflow.log_metrics({k: v for k, v in lr_m.items() if isinstance(v, float)})
        mlflow.sklearn.log_model(lr, "model")
        results.append(("logistic_baseline", lr, lr_m))
        logger.info(f"  Baseline  → Gini: {lr_m['gini']} | AUC: {lr_m['auc_roc']}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost_champion"):
        logger.info("Entrenando XGBoost...")
        xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
        xgb_m     = evaluate(xgb_model, X_test, y_test, "xgboost")
        mlflow.log_metrics({k: v for k, v in xgb_m.items() if isinstance(v, float)})
        mlflow.sklearn.log_model(xgb_model, "model")
        results.append(("xgboost", xgb_model, xgb_m))
        logger.info(f"  XGBoost   → Gini: {xgb_m['gini']} | AUC: {xgb_m['auc_roc']}")

    # ── Selección de champion ─────────────────────────────────────────────────
    champion_name, champion_model, champion_metrics = max(
        results, key=lambda x: x[2]["gini"]
    )
    baseline_gini = next(m for n, _, m in results if n == "logistic_baseline")["gini"]
    champion_metrics["gini_lift_vs_baseline"] = round(
        champion_metrics["gini"] - baseline_gini, 4
    )

    logger.success(
        f"Champion: {champion_name} | "
        f"Gini: {champion_metrics['gini']} | "
        f"Lift: +{champion_metrics['gini_lift_vs_baseline']}"
    )

    # Guardar artefactos
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(champion_model, f)
    with open(MODELS_DIR / "baseline.pkl", "wb") as f:
        pickle.dump(lr, f)

    metadata = {
        "champion":         champion_name,
        "champion_metrics": champion_metrics,
        "all_models":       [m for _, _, m in results],
        "feature_count":    int(X_train.shape[1]),
        "train_rows":       int(len(X_train)),
        "test_rows":        int(len(X_test)),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open("reports/train_metrics.json", "w") as f:
        json.dump(champion_metrics, f, indent=2)

    logger.success("Training OK — artefactos guardados en models/champion/")


if __name__ == "__main__":
    train()
