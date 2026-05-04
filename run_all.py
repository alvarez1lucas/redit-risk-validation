"""
run_all.py — Pipeline completo Home Credit Default Risk
=======================================================
Corre las 7 etapas en orden.
Prerequisito: data/raw/application_train.csv

Uso:
    python run_all.py

Para la API después del pipeline:
    uvicorn src.api.main:app --reload --port 8000
"""
from pathlib import Path
from loguru import logger


def main():
    logger.info("=" * 60)
    logger.info("Home Credit — Model Validation Pipeline")
    logger.info("=" * 60)

    # Verificar que el CSV existe antes de arrancar
    csv_path = Path("data/raw/application_train.csv")
    if not csv_path.exists():
        logger.error(
            f"No se encontró {csv_path}\n"
            "Descargá application_train.csv de Kaggle:\n"
            "  https://www.kaggle.com/c/home-credit-default-risk/data\n"
            "y colocalo en data/raw/"
        )
        return

    # Crear carpetas de output
    for d in ["data/processed", "models/champion",
              "reports/figures", "reports/model_cards"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    logger.info("\n[1/7] Ingesta de datos...")
    from src.data.ingest import ingest
    ingest()

    # ── Stage 2: Features ─────────────────────────────────────────────────────
    logger.info("\n[2/7] Feature engineering...")
    from src.features.build import build
    build()

    # ── Stage 3: Train ────────────────────────────────────────────────────────
    logger.info("\n[3/7] Entrenamiento...")
    from src.models.train import train
    train()

    # ── Stage 4: SR 11-7 Validation ───────────────────────────────────────────
    logger.info("\n[4/7] Validación SR 11-7...")
    from src.validation.sr117 import validate
    validate()

    # ── Stage 5: Fairness ─────────────────────────────────────────────────────
    logger.info("\n[5/7] Análisis de fairness...")
    from src.governance.fairness import compute_fairness
    compute_fairness()

    # ── Stage 6: SHAP ─────────────────────────────────────────────────────────
    logger.info("\n[6/7] Explainability (SHAP)...")
    from src.validation.explain import explain
    explain()

    # ── Stage 7: Drift baseline ───────────────────────────────────────────────
    logger.info("\n[7/7] Drift monitoring baseline...")
    from src.monitoring.baseline import create_baseline
    create_baseline()

    # ── Model Card ────────────────────────────────────────────────────────────
    logger.info("\nGenerando Model Card...")
    from src.governance.model_card import generate_model_card
    generate_model_card()

    logger.success("\n" + "=" * 60)
    logger.success("Pipeline completo!")
    logger.success("=" * 60)
    logger.info("\nOutputs generados:")
    logger.info("  models/champion/model.pkl             — modelo champion")
    logger.info("  models/champion/shap_explainer.pkl    — SHAP explainer")
    logger.info("  reports/validation_metrics.json       — métricas SR 11-7")
    logger.info("  reports/fairness_metrics.json         — métricas fairness")
    logger.info("  reports/model_cards/model_card.html   — model card completo")
    logger.info("  reports/figures/                      — gráficos")
    logger.info("\nPróximos pasos:")
    logger.info("  1. Abrí notebooks/01_eda.ipynb y completá el análisis")
    logger.info("  2. uvicorn src.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
