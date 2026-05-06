"""Tests for drift monitoring module."""
import pytest
import numpy as np
from src.monitoring import PredictionMonitor


@pytest.fixture
def monitor():
    return PredictionMonitor(window_size=500)


def test_check_drift_without_reference(monitor):
    monitor.record(0.7)
    result = monitor.check_drift()
    assert result["drift_detected"] is False


def test_check_drift_with_insufficient_samples(monitor):
    monitor.set_reference([0.5] * 100)
    for _ in range(10):
        monitor.record(0.9)
    result = monitor.check_drift()
    assert result["drift_detected"] is False


def test_no_drift_with_similar_distributions(monitor):
    rng = np.random.default_rng(42)
    ref = rng.uniform(0.4, 0.7, 200).tolist()
    monitor.set_reference(ref)
    for _ in range(100):
        monitor.record(float(rng.uniform(0.4, 0.7)))
    result = monitor.check_drift()
    assert result["ks_statistic"] >= 0.0


def test_drift_detected_with_different_distributions(monitor):
    rng = np.random.default_rng(42)
    ref = rng.uniform(0.2, 0.4, 300).tolist()
    monitor.set_reference(ref)
    for _ in range(150):
        monitor.record(float(rng.uniform(0.7, 1.0)))
    result = monitor.check_drift()
    assert result["ks_statistic"] > 0.0
    assert "drift_detected" in result


def test_summary_empty(monitor):
    summary = monitor.summary()
    assert summary["count"] == 0
    assert summary["mean"] is None


def test_summary_with_data(monitor):
    for v in [0.3, 0.5, 0.7, 0.9]:
        monitor.record(v)
    summary = monitor.summary()
    assert summary["count"] == 4
    assert 0.0 < summary["mean"] < 1.0
    assert summary["std"] >= 0.0


def test_reset_clears_predictions(monitor):
    monitor.record(0.5)
    monitor.record(0.6)
    monitor.reset()
    assert len(monitor.predictions) == 0
    summary = monitor.summary()
    assert summary["count"] == 0


def test_window_size_respected():
    m = PredictionMonitor(window_size=10)
    for i in range(20):
        m.record(float(i) / 20)
    assert len(m.predictions) == 10


@pytest.mark.parametrize("prob", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_record_boundary_values(monitor, prob):
    monitor.record(prob)
    assert len(monitor.predictions) == 1
    assert monitor.predictions[0] == prob
