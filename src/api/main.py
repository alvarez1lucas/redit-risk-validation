"""
FastAPI — Credit Risk Scoring API
===================================
POST /predict       → score + SHAP explanation
GET  /model/health  → health check
GET  /model/info    → model metadata
"""
import json
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

MODELS_DIR = Path("models/champion")
DATA_DIR   = Path("data/processed")

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando modelo...")
    with open(MODELS_DIR / "model.pkl",           "rb") as f: _state["model"]        = pickle.load(f)
    with open(MODELS_DIR / "shap_explainer.pkl",  "rb") as f: _state["explainer"]    = pickle.load(f)
    with open(DATA_DIR   / "preprocessor.pkl",    "rb") as f: _state["preprocessor"] = pickle.load(f)
    with open(DATA_DIR   / "feature_names.json")        as f: _state["feature_names"]= json.load(f)
    with open(MODELS_DIR / "model_metadata.json")       as f: _state["metadata"]     = json.load(f)
    _state["request_count"] = 0
    _state["start_time"]    = time.time()
    logger.success("API lista")
    yield
    _state.clear()


app = FastAPI(
    title="Home Credit Risk Scoring API",
    description="SR 11-7 compliant credit scoring con SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CreditApplication(BaseModel):
    AMT_CREDIT:                     float = Field(..., gt=0,  example=270000)
    AMT_ANNUITY:                    float = Field(..., gt=0,  example=13500)
    AMT_INCOME_TOTAL:               float = Field(..., gt=0,  example=135000)
    AMT_GOODS_PRICE:                float = Field(None,       example=270000)
    DAYS_BIRTH:                     int   = Field(...,        example=-12000)
    DAYS_EMPLOYED:                  int   = Field(None,       example=-3000)
    EXT_SOURCE_1:                   float = Field(None,       example=0.50)
    EXT_SOURCE_2:                   float = Field(None,       example=0.60)
    EXT_SOURCE_3:                   float = Field(None,       example=0.55)
    NAME_CONTRACT_TYPE:             str   = Field(...,        example="Cash loans")
    NAME_INCOME_TYPE:               str   = Field(...,        example="Working")
    NAME_EDUCATION_TYPE:            str   = Field(...,        example="Secondary / secondary special")
    NAME_FAMILY_STATUS:             str   = Field(...,        example="Married")
    NAME_HOUSING_TYPE:              str   = Field(...,        example="House / apartment")
    OCCUPATION_TYPE:                str   = Field(None,       example="Laborers")
    CODE_GENDER:                    str   = Field(None,       example="M",
                                                  description="Solo para fairness monitoring")


class PredictionResponse(BaseModel):
    decision:               str
    probability_of_default: float
    risk_score:             float
    confidence:             str
    explanation:            dict
    model_version:          str
    latency_ms:             float


@app.get("/model/health")
async def health():
    return {
        "status":           "healthy",
        "uptime_seconds":   round(time.time() - _state.get("start_time", time.time()), 1),
        "requests_served":  _state.get("request_count", 0),
    }


@app.get("/model/info")
async def model_info():
    return _state.get("metadata", {})


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    t0 = time.time()
    _state["request_count"] = _state.get("request_count", 0) + 1

    try:
        data = application.model_dump(exclude={"CODE_GENDER"})
        df   = pd.DataFrame([data])

        preprocessor  = _state["preprocessor"]
        model         = _state["model"]
        explainer     = _state["explainer"]
        feature_names = _state["feature_names"]

        X    = preprocessor.transform(df)
        X_df = pd.DataFrame(X, columns=feature_names)

        proba    = float(model.predict_proba(X_df)[0, 1])
        decision = "approved" if proba < 0.50 else "rejected"

        # SHAP explanation
        try:
            sv = explainer.shap_values(X_df)
            if isinstance(sv, list): sv = sv[1]
            sv = sv.flatten()
            top = sorted(zip(feature_names, sv), key=lambda x: -abs(x[1]))[:5]
            explanation = {
                "top_factors": [
                    {"feature":   f.replace("num__","").replace("cat__",""),
                     "impact":    round(float(v), 4),
                     "direction": "increases_risk" if v > 0 else "reduces_risk"}
                    for f, v in top
                ]
            }
        except Exception:
            explanation = {"top_factors": [], "note": "explanation_unavailable"}

        conf = ("high"   if proba < 0.20 or proba > 0.80
                else "medium" if proba < 0.35 or proba > 0.65
                else "low")

        return PredictionResponse(
            decision               = decision,
            probability_of_default = round(proba, 4),
            risk_score             = round(1 - proba, 4),
            confidence             = conf,
            explanation            = explanation,
            model_version          = _state["metadata"].get("champion", "unknown"),
            latency_ms             = round((time.time() - t0) * 1000, 2),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
