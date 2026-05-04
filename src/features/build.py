"""
Feature engineering — Home Credit Default Risk
===============================================
Lee train.parquet y test.parquet generados por ingest.py.
Aplica el preprocessor (scaler + OHE) sin data leakage.
Guarda X_train, X_test, y_train, y_test y el preprocessor.
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PROCESSED = Path("data/processed")


def load_metadata() -> dict:
    with open(DATA_PROCESSED / "metadata.json") as f:
        return json.load(f)


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Pipeline de transformación:
      - Numéricos: imputar con mediana (maneja los muchos NaN de Home Credit) + StandardScaler
      - Categóricos: imputar con 'missing' + OneHotEncoder
    Se fitea SOLO sobre train para evitar data leakage.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      numeric_features),
            ("cat", categorical_pipeline,  categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build():
    logger.info("=" * 50)
    logger.info("STAGE 2/7 — Features")
    logger.info("=" * 50)

    meta = load_metadata()

    train_df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    test_df  = pd.read_parquet(DATA_PROCESSED / "test.parquet")

    # Columnas base + derivadas
    numeric_features = (
        meta["numeric_features"]
        + [f for f in meta.get("derived_features", [])
           if f in train_df.columns and f not in ["age_years"]]
    )
    # age_years ya está en el df como numérica derivada
    if "age_years" in train_df.columns:
        numeric_features = numeric_features + ["age_years"]

    categorical_features = [
        c for c in meta["categorical_features"] if c in train_df.columns
    ]

    # Eliminar duplicados manteniendo orden
    numeric_features     = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))

    # Separar target — protegidos se quedan en el raw para fairness
    target_col = meta["target"]
    y_train = train_df[target_col]
    y_test  = test_df[target_col]

    # Verificar que las features existen en el dataframe
    numeric_features     = [c for c in numeric_features     if c in train_df.columns]
    categorical_features = [c for c in categorical_features if c in train_df.columns]

    logger.info(f"Features numéricas:    {len(numeric_features)}")
    logger.info(f"Features categóricas:  {len(categorical_features)}")

    # Fit preprocessor SOLO sobre train
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_arr  = preprocessor.fit_transform(train_df)
    X_test_arr   = preprocessor.transform(test_df)

    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)

    # Guardar artefactos
    X_train.to_parquet(DATA_PROCESSED / "X_train.parquet", index=False)
    X_test.to_parquet(DATA_PROCESSED  / "X_test.parquet",  index=False)
    y_train.reset_index(drop=True).to_frame().to_parquet(DATA_PROCESSED / "y_train.parquet", index=False)
    y_test.reset_index(drop=True).to_frame().to_parquet(DATA_PROCESSED  / "y_test.parquet",  index=False)

    with open(DATA_PROCESSED / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(DATA_PROCESSED / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    logger.success(
        f"Features OK — {X_train.shape[1]} features | "
        f"train {X_train.shape[0]:,} | test {X_test.shape[0]:,}"
    )


if __name__ == "__main__":
    build()
