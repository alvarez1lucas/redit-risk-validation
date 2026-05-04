"""
Tests del pipeline — Home Credit Default Risk
=============================================
Correr con: pytest tests/ -v
"""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.validation.sr117 import (
    compute_ks, compute_psi, gini, hosmer_lemeshow
)
from src.governance.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds,
)


@pytest.fixture
def good_predictions():
    np.random.seed(42)
    y_true  = np.array([0]*200 + [1]*200)
    y_score = np.concatenate([
        np.random.beta(2, 5, 200),
        np.random.beta(5, 2, 200),
    ])
    return y_true, y_score


@pytest.fixture
def random_predictions():
    np.random.seed(42)
    return np.random.randint(0, 2, 500), np.random.uniform(0, 1, 500)


class TestDiscriminatoryPower:
    def test_gini_good_model(self, good_predictions):
        y, s = good_predictions
        assert gini(y, s) > 0.40

    def test_gini_random_model(self, random_predictions):
        y, s = random_predictions
        assert abs(gini(y, s)) < 0.15

    def test_ks_range(self, good_predictions):
        y, s = good_predictions
        ks = compute_ks(y, s)
        assert 0 <= ks["ks_statistic"] <= 1
        assert 0 <= ks["ks_pvalue"]    <= 1


class TestCalibration:
    def test_hl_output_keys(self, good_predictions):
        y, s = good_predictions
        hl = hosmer_lemeshow(y, s)
        assert "hl_statistic"   in hl
        assert "hl_pvalue"      in hl
        assert "well_calibrated" in hl

    def test_hl_pvalue_range(self, good_predictions):
        y, s = good_predictions
        hl = hosmer_lemeshow(y, s)
        assert 0 <= hl["hl_pvalue"] <= 1


class TestStability:
    def test_psi_identical(self):
        d = np.random.normal(0, 1, 1000)
        assert compute_psi(d, d) < 0.01

    def test_psi_different(self):
        e = np.random.normal(0, 1, 1000)
        a = np.random.normal(3, 1, 1000)
        assert compute_psi(e, a) > 0.25


class TestFairness:
    def test_dpd_equal_groups(self):
        y_app = np.array([1]*100 + [0]*100 + [1]*100 + [0]*100)
        group = np.array(["M"]*200 + ["F"]*200)
        dpd   = demographic_parity_difference(y_app, group)
        assert abs(dpd) < 0.05

    def test_dir_unequal_groups(self):
        y_app = np.array([1]*160 + [0]*40 + [1]*40 + [0]*160)
        group = np.array(["M"]*200 + ["F"]*200)
        dir_r = disparate_impact_ratio(y_app, group)
        assert dir_r < 0.80

    def test_equalized_odds_keys(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred = np.random.randint(0, 2, 200)
        group  = np.array(["A"]*100 + ["B"]*100)
        eo     = equalized_odds(y_true, y_pred, group)
        assert "A" in eo and "B" in eo
        assert "tpr_gap" in eo and "fpr_gap" in eo
