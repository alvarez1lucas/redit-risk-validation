"""
SR 11-7 Validation Suite — Home Credit Default Risk
=====================================================
Tests requeridos por la guía de Model Risk Management de la Fed:
  1. Poder discriminatorio: Gini, AUC-ROC, KS
  2. Calibración: Hosmer-Lemeshow
  3. Estabilidad: PSI (train vs test)
  4. Stress testing con shocks financieros
  5. Análisis de sensibilidad (permutation importance)
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

MODELS_DIR  = Path("models/champion")
DATA_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def gini(y_true, y_score) -> float:
    return round(2 * roc_auc_score(y_true, y_score) - 1, 4)


def compute_ks(y_true, y_score) -> dict:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    ks_stat, ks_p = stats.ks_2samp(neg, pos)
    return {"ks_statistic": round(float(ks_stat), 4), "ks_pvalue": round(float(ks_p), 4)}


def hosmer_lemeshow(y_true, y_score, n_bins=10) -> dict:
    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["decile"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        obs_events=("y", "sum"),
        total=("y", "count"),
        mean_pred=("p", "mean"),
    )
    g["exp_events"]    = g["total"] * g["mean_pred"]
    g["exp_nonevents"] = g["total"] * (1 - g["mean_pred"])
    g["obs_nonevents"] = g["total"] - g["obs_events"]
    hl = (
        ((g["obs_events"]    - g["exp_events"])    ** 2 / g["exp_events"].clip(1e-6)).sum()
        + ((g["obs_nonevents"] - g["exp_nonevents"]) ** 2 / g["exp_nonevents"].clip(1e-6)).sum()
    )
    p = 1 - stats.chi2.cdf(hl, n_bins - 2)
    return {
        "hl_statistic":   round(float(hl), 4),
        "hl_pvalue":      round(float(p), 4),
        "well_calibrated": bool(p > 0.05),
    }


def compute_psi(expected, actual, n_bins=10) -> float:
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6; bins[-1] += 1e-6
    def bucket(d):
        c, _ = np.histogram(d, bins=bins)
        p = c / len(d)
        return np.where(p == 0, 1e-4, p)
    e, a = bucket(expected), bucket(actual)
    return round(float(np.sum((a - e) * np.log(a / e))), 4)


def stress_test(model, X_test, y_test) -> dict:
    """
    Escenarios de stress calibrados con contexto de mercados emergentes.
    Los multiplicadores se basan en variaciones observadas en crisis reales.
    """
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    scenarios = {
        "credit_expansion_mild": {
            "description": "Expansión crediticia leve: monto +20%, cuota +15%",
            "cols": {"num__AMT_CREDIT": 1.20, "num__AMT_ANNUITY": 1.15},
        },
        "income_shock_moderate": {
            "description": "Shock de ingreso moderado: ingreso -25% (desempleo/inflación)",
            "cols": {"num__AMT_INCOME_TOTAL": 0.75},
        },
        "income_shock_severe": {
            "description": "Shock de ingreso severo: ingreso -40%, crédito +30%",
            "cols": {"num__AMT_INCOME_TOTAL": 0.60, "num__AMT_CREDIT": 1.30},
        },
        "bureau_score_deterioration": {
            "description": "Deterioro de bureau: EXT_SOURCE -20% (crisis crediticia sistémica)",
            "cols": {
                "num__EXT_SOURCE_1": 0.80,
                "num__EXT_SOURCE_2": 0.80,
                "num__EXT_SOURCE_3": 0.80,
            },
        },
    }

    results = {"baseline_auc": round(float(base_auc), 4), "baseline_gini": gini(y_test, model.predict_proba(X_test)[:, 1]), "scenarios": {}}
    for name, sc in scenarios.items():
        X_s = X_test.copy()
        for col, mult in sc["cols"].items():
            if col in X_s.columns:
                X_s[col] = X_s[col] * mult
        s_auc = roc_auc_score(y_test, model.predict_proba(X_s)[:, 1])
        results["scenarios"][name] = {
            "description":    sc["description"],
            "auc":            round(float(s_auc), 4),
            "gini":           round(2 * float(s_auc) - 1, 4),
            "auc_degradation": round(float(base_auc - s_auc), 4),
        }
        logger.info(f"  Stress [{name}]: AUC {s_auc:.4f} (Δ -{base_auc - s_auc:.4f})")
    return results


def sensitivity_analysis(model, X_test, y_test, top_n=15) -> dict:
    """Permutation importance — cuánto cae el AUC al permutar cada feature."""
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    imp = {}
    for feat in X_test.columns[:top_n]:
        X_p = X_test.copy()
        X_p[feat] = np.random.permutation(X_p[feat].values)
        imp[feat] = round(float(base_auc - roc_auc_score(y_test, model.predict_proba(X_p)[:, 1])), 5)
    return dict(sorted(imp.items(), key=lambda x: -x[1]))


def plot_roc(y_true, y_score, baseline_score):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    axes[0].plot(fpr, tpr, color="#534AB7", linewidth=2, label=f"Champion (AUC={auc:.3f})")
    if baseline_score is not None:
        fb, tb, _ = roc_curve(y_true, baseline_score)
        ab = roc_auc_score(y_true, baseline_score)
        axes[0].plot(fb, tb, color="#888780", linewidth=1.5, linestyle="--", label=f"Baseline (AUC={ab:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.5)
    axes[0].set_title("ROC Curve — SR 11-7"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")

    thresholds = np.linspace(0, 1, 100)
    cum_pos = [(y_score[y_true == 1] <= t).mean() for t in thresholds]
    cum_neg = [(y_score[y_true == 0] <= t).mean() for t in thresholds]
    ks_vals = np.abs(np.array(cum_pos) - np.array(cum_neg))
    ks_t = thresholds[np.argmax(ks_vals)]
    axes[1].plot(thresholds, cum_pos, color="#D85A30", linewidth=2, label="Defaulters")
    axes[1].plot(thresholds, cum_neg, color="#1D9E75", linewidth=2, label="Non-defaulters")
    axes[1].axvline(ks_t, color="#534AB7", linestyle="--", linewidth=1.5, label=f"KS={ks_vals.max():.3f}")
    axes[1].set_title("KS Statistic"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Score"); axes[1].set_ylabel("CDF")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_ks.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true, y_score):
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfecta")
    plt.plot(prob_pred, prob_true, "o-", color="#534AB7", linewidth=2, label="Modelo")
    plt.title("Calibración (Reliability Diagram)")
    plt.xlabel("Probabilidad predicha"); plt.ylabel("Fracción de positivos")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def validate():
    logger.info("=" * 50)
    logger.info("STAGE 4/7 — SR 11-7 Validation")
    logger.info("=" * 50)

    with open(MODELS_DIR / "model.pkl",    "rb") as f: model    = pickle.load(f)
    with open(MODELS_DIR / "baseline.pkl", "rb") as f: baseline = pickle.load(f)

    X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test  = pd.read_parquet(DATA_DIR / "y_test.parquet").iloc[:, 0]
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")

    y_score_champ = model.predict_proba(X_test)[:, 1]
    y_score_base  = baseline.predict_proba(X_test)[:, 1]
    y_score_train = model.predict_proba(X_train)[:, 1]

    gini_champ = gini(y_test, y_score_champ)
    gini_base  = gini(y_test, y_score_base)
    auc        = round(float(roc_auc_score(y_test, y_score_champ)), 4)
    ks         = compute_ks(y_test.values, y_score_champ)
    hl         = hosmer_lemeshow(y_test.values, y_score_champ)
    psi        = compute_psi(y_score_train, y_score_champ)
    psi_status = "stable" if psi < 0.10 else "monitor" if psi < 0.25 else "retrain_required"
    stress     = stress_test(model, X_test, y_test)
    sensitivity = sensitivity_analysis(model, X_test, y_test)

    logger.info(f"  Gini:     {gini_champ} (baseline: {gini_base}, lift: +{round(gini_champ-gini_base,4)})")
    logger.info(f"  AUC-ROC:  {auc}")
    logger.info(f"  KS:       {ks['ks_statistic']}")
    logger.info(f"  HL p:     {hl['hl_pvalue']} → {'calibrado' if hl['well_calibrated'] else 'DESCALIBRADO'}")
    logger.info(f"  PSI:      {psi} ({psi_status})")

    thresholds = {"min_gini": 0.20, "min_ks": 0.15, "max_psi": 0.25}
    sr117_pass = all([
        gini_champ >= thresholds["min_gini"],
        ks["ks_statistic"] >= thresholds["min_ks"],
        psi <= thresholds["max_psi"],
    ])

    report = {
        "sr117_overall_pass": sr117_pass,
        "discriminatory_power": {
            "gini":              gini_champ,
            "auc_roc":           auc,
            "gini_lift_baseline": round(gini_champ - gini_base, 4),
            **ks,
        },
        "calibration":  hl,
        "stability":    {"psi": psi, "psi_status": psi_status},
        "stress_testing": stress,
        "sensitivity_top10": dict(list(sensitivity.items())[:10]),
        "thresholds_applied": thresholds,
    }

    with open(REPORTS_DIR / "sr117_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(REPORTS_DIR / "validation_metrics.json", "w") as f:
        json.dump({
            "gini":       gini_champ,
            "auc_roc":    auc,
            "ks":         ks["ks_statistic"],
            "psi":        psi,
            "sr117_pass": int(sr117_pass),
        }, f, indent=2)

    plot_roc(y_test.values, y_score_champ, y_score_base)
    plot_calibration(y_test.values, y_score_champ)

    logger.success(f"SR 11-7: {'PASSED' if sr117_pass else 'FAILED'}")
    return report


if __name__ == "__main__":
    validate()
