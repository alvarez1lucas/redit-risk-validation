"""
Fairness & Bias Analysis — Home Credit Default Risk
====================================================
Atributos protegidos analizados:
  - CODE_GENDER (género)
  - age_years agrupado en buckets (edad)

Métricas: Demographic Parity Difference, Disparate Impact Ratio,
          Equalized Odds, AUC por grupo.

Marcos regulatorios: ECOA, EU AI Act Art.10, BCRA Com. A 7724
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score

MODELS_DIR  = Path("models/champion")
DATA_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def demographic_parity_difference(y_approved, group) -> float:
    groups = np.unique(group)
    if len(groups) != 2:
        return float("nan")
    r0 = y_approved[group == groups[0]].mean()
    r1 = y_approved[group == groups[1]].mean()
    return round(float(r0 - r1), 4)


def disparate_impact_ratio(y_approved, group) -> float:
    groups = np.unique(group)
    rates  = [y_approved[group == g].mean() for g in groups]
    if max(rates) == 0:
        return float("nan")
    return round(float(min(rates) / max(rates)), 4)


def equalized_odds(y_true, y_pred, group) -> dict:
    groups = np.unique(group)
    result = {}
    for g in groups:
        m = group == g
        if m.sum() < 30:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[m], y_pred[m], labels=[0, 1]).ravel()
        result[str(g)] = {
            "tpr": round(float(tp / (tp + fn + 1e-9)), 4),
            "fpr": round(float(fp / (fp + tn + 1e-9)), 4),
        }
    if len(groups) == 2:
        g0, g1 = str(groups[0]), str(groups[1])
        if g0 in result and g1 in result:
            result["tpr_gap"] = round(abs(result[g0]["tpr"] - result[g1]["tpr"]), 4)
            result["fpr_gap"] = round(abs(result[g0]["fpr"] - result[g1]["fpr"]), 4)
    return result


def group_auc(y_true, y_score, group) -> dict:
    result = {}
    for g in np.unique(group):
        m = group == g
        if m.sum() >= 30 and y_true[m].nunique() == 2:
            result[str(g)] = round(float(roc_auc_score(y_true[m], y_score[m])), 4)
        else:
            result[str(g)] = None
    return result


def analyze_attribute(model, X_test, y_test, y_score, group_arr,
                       attr_name, threshold=0.50) -> dict:
    y_pred     = (y_score >= threshold).astype(int)
    y_approved = (y_pred == 0).astype(int)

    groups_unique = np.unique(group_arr)
    approval_rates = {
        str(g): round(float(y_approved[group_arr == g].mean()), 4)
        for g in groups_unique
    }

    dpd    = demographic_parity_difference(y_approved, group_arr)
    dir_r  = disparate_impact_ratio(y_approved, group_arr)
    eq_odd = equalized_odds(y_test.values, y_pred, group_arr)
    aucs   = group_auc(y_test, y_score, group_arr)

    flags = []
    if not np.isnan(dpd) and abs(dpd) > 0.10:
        flags.append(f"DPD {dpd:.3f} supera ±0.10")
    if not np.isnan(dir_r) and dir_r < 0.80:
        flags.append(f"DIR {dir_r:.3f} por debajo de 0.80 (riesgo ECOA)")
    if eq_odd.get("tpr_gap", 0) > 0.10:
        flags.append(f"TPR gap {eq_odd['tpr_gap']:.3f} supera 0.10")
    if eq_odd.get("fpr_gap", 0) > 0.10:
        flags.append(f"FPR gap {eq_odd['fpr_gap']:.3f} supera 0.10")

    for flag in flags:
        logger.warning(f"  [FAIRNESS] {attr_name}: {flag}")
    if not flags:
        logger.info(f"  {attr_name}: OK — todos los checks de fairness pasan")

    return {
        "approval_rates":               approval_rates,
        "demographic_parity_difference": dpd,
        "disparate_impact_ratio":        dir_r,
        "equalized_odds":                eq_odd,
        "auc_by_group":                  aucs,
        "regulatory_flags":              flags,
        "passed":                        len(flags) == 0,
    }


def plot_fairness(fairness_results: dict):
    n = len(fairness_results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    for idx, (attr, res) in enumerate(fairness_results.items()):
        ax_bar, ax_eqo = axes[idx]

        groups = list(res["approval_rates"].keys())
        rates  = [res["approval_rates"][g] for g in groups]
        colors = ["#534AB7", "#D85A30", "#1D9E75", "#888780"]
        bars   = ax_bar.bar(groups, rates, color=colors[:len(groups)], alpha=0.85, edgecolor="white")
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title(f"{attr}: tasa de aprobación por grupo")
        ax_bar.set_ylabel("Tasa de aprobación")
        ax_bar.axhline(np.mean(rates), color="gray", linestyle="--", linewidth=1, label="Promedio")
        ax_bar.legend(fontsize=9)
        for bar, rate in zip(bars, rates):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{rate:.1%}", ha="center", fontsize=9)

        eq = res.get("equalized_odds", {})
        gkeys = [k for k in eq if k not in ("tpr_gap", "fpr_gap")]
        if gkeys:
            x = np.arange(len(gkeys))
            ax_eqo.bar(x - 0.18, [eq[k]["tpr"] for k in gkeys], 0.35,
                       label="TPR", color="#1D9E75", alpha=0.85)
            ax_eqo.bar(x + 0.18, [eq[k]["fpr"] for k in gkeys], 0.35,
                       label="FPR", color="#D85A30", alpha=0.85)
            ax_eqo.set_xticks(x)
            ax_eqo.set_xticklabels(gkeys)
            ax_eqo.set_ylim(0, 1)
            ax_eqo.set_title(f"{attr}: Equalized Odds")
            ax_eqo.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Fairness dashboard guardado")


def compute_fairness(threshold: float = 0.50):
    logger.info("=" * 50)
    logger.info("STAGE 5/7 — Fairness Analysis")
    logger.info("=" * 50)

    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test   = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test   = pd.read_parquet(DATA_DIR / "y_test.parquet").iloc[:, 0]
    raw_test = pd.read_parquet(DATA_DIR / "test.parquet")

    y_score = model.predict_proba(X_test)[:, 1]
    fairness_results = {}

    # ── Género ────────────────────────────────────────────────────────────────
    if "CODE_GENDER" in raw_test.columns:
        gender = raw_test["CODE_GENDER"].fillna("Unknown").values
        # Filtrar solo M y F (excluir XNA)
        mask   = np.isin(gender, ["M", "F"])
        if mask.sum() > 100:
            fairness_results["gender"] = analyze_attribute(
                model, X_test[mask], y_test[mask],
                y_score[mask], gender[mask], "gender", threshold
            )

    # ── Edad ──────────────────────────────────────────────────────────────────
    age_col = "age_years" if "age_years" in raw_test.columns else None
    if age_col:
        age_group = pd.cut(
            raw_test[age_col],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["<25", "25-35", "35-45", "45-55", ">55"]
        ).astype(str).values
        fairness_results["age_group"] = analyze_attribute(
            model, X_test, y_test, y_score, age_group, "age_group", threshold
        )

    overall_pass = all(v["passed"] for v in fairness_results.values())

    report = {
        "overall_fairness_passed":      overall_pass,
        "threshold_used":               threshold,
        "protected_attributes_analyzed": list(fairness_results.keys()),
        "results":                      fairness_results,
        "thresholds": {
            "demographic_parity_max_abs": 0.10,
            "disparate_impact_min":       0.80,
            "equalized_odds_max_gap":     0.10,
        },
        "regulatory_frameworks": ["ECOA", "EU AI Act Art.10", "BCRA Com. A 7724"],
    }

    with open(REPORTS_DIR / "fairness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    metrics = {"fairness_pass": int(overall_pass)}
    if "gender" in fairness_results:
        metrics["gender_dpd"] = fairness_results["gender"]["demographic_parity_difference"]
        metrics["gender_dir"] = fairness_results["gender"]["disparate_impact_ratio"]
    if "age_group" in fairness_results:
        metrics["age_dpd"] = fairness_results["age_group"]["demographic_parity_difference"]

    with open(REPORTS_DIR / "fairness_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_fairness(fairness_results)
    logger.success(f"Fairness: {'PASSED' if overall_pass else 'FAILED — revisar flags'}")
    return report


if __name__ == "__main__":
    compute_fairness()
