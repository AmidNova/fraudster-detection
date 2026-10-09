import numpy as np
import pytest

from fraud_detection.evaluation import (
    alert_stats,
    cost_optimal_threshold,
    expected_cost,
    population_stability_index,
)

Y_TRUE = np.array([0, 0, 0, 1, 1])
SCORES = np.array([0.1, 0.2, 0.6, 0.4, 0.9])


def test_alert_stats():
    stats = alert_stats(Y_TRUE, SCORES >= 0.5)
    assert stats == {"n_flagged": 2, "fraud_caught": 1, "precision": 0.5, "recall": 0.5}


def test_alert_stats_with_no_alert_has_zero_precision():
    assert alert_stats(Y_TRUE, np.zeros(5, dtype=bool))["precision"] == 0


def test_expected_cost_weights_missed_fraudsters():
    # threshold 0.5: 1 false alarm, 1 missed fraudster
    assert expected_cost(Y_TRUE, SCORES, 0.5, ratio=10) == 1 + 10


def test_cost_optimal_threshold_lowers_threshold_when_misses_are_expensive():
    grid = np.linspace(0.05, 0.95, 19)
    cheap_miss = cost_optimal_threshold(Y_TRUE, SCORES, ratio=0.5, grid=grid)
    costly_miss = cost_optimal_threshold(Y_TRUE, SCORES, ratio=10, grid=grid)
    assert costly_miss < cheap_miss
    assert costly_miss <= 0.4   # every fraudster flagged


class TestPopulationStabilityIndex:
    def test_same_distribution_is_stable(self):
        rng = np.random.default_rng(0)
        assert population_stability_index(rng.normal(size=5000), rng.normal(size=5000)) < 0.1

    def test_shifted_distribution_is_significant(self):
        rng = np.random.default_rng(0)
        assert population_stability_index(rng.normal(size=5000), rng.normal(1, size=5000)) > 0.25

    def test_constant_reference_does_not_crash(self):
        assert population_stability_index(np.zeros(100), np.zeros(100)) == pytest.approx(0, abs=1e-9)
        assert population_stability_index(np.zeros(100), np.ones(100)) > 0.25

    def test_mostly_zero_feature(self):
        reference = np.r_[np.zeros(95), np.arange(5)]
        assert population_stability_index(reference, reference) == pytest.approx(0, abs=1e-9)
