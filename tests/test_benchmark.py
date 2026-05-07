"""Tests for performance benchmark utilities."""
import pytest

from scripts.benchmark import benchmark_eligibility_matcher, benchmark_ml_predictor, time_function


def test_time_function_basic():
    def add(a, b):
        return a + b
    result, elapsed = time_function(add, 1, 2)
    assert result == 3
    assert elapsed >= 0.0


def test_benchmark_eligibility_returns_expected_keys():
    result = benchmark_eligibility_matcher(n_patients=5, n_trials=2)
    assert "total_pairs" in result
    assert "total_seconds" in result
    assert "pairs_per_second" in result
    assert "mean_ms" in result
    assert result["total_pairs"] == 10


def test_benchmark_eligibility_positive_throughput():
    result = benchmark_eligibility_matcher(n_patients=10, n_trials=5)
    assert result["pairs_per_second"] > 0
    assert result["mean_ms"] > 0


def test_benchmark_ml_predictor_returns_expected_keys():
    result = benchmark_ml_predictor(n_patients=5)
    assert "total_predictions" in result
    assert "predictions_per_second" in result
    assert result["total_predictions"] == 5


@pytest.mark.parametrize("n_patients,n_trials", [(5, 2), (10, 1), (20, 5)])
def test_eligibility_benchmark_pair_count(n_patients, n_trials):
    result = benchmark_eligibility_matcher(n_patients, n_trials)
    assert result["total_pairs"] == n_patients * n_trials
