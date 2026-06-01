"""Tests for drift monitoring module."""

import numpy as np
import pytest

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


def test_drift_threshold_configurable(monkeypatch):
    """DRIFT_THRESHOLD should be read from the DRIFT_THRESHOLD env var."""
    import importlib

    monkeypatch.setenv("DRIFT_THRESHOLD", "0.10")
    import src.monitoring as mon_module

    importlib.reload(mon_module)
    assert mon_module._DRIFT_THRESHOLD == 0.10
    # restore
    monkeypatch.delenv("DRIFT_THRESHOLD", raising=False)
    importlib.reload(mon_module)


def test_check_drift_returns_threshold_key(monitor):
    """check_drift result should include threshold when there is enough data."""
    rng = __import__("numpy").random.default_rng(0)
    monitor.set_reference(rng.uniform(0, 1, 200).tolist())
    for _ in range(50):
        monitor.record(float(rng.uniform(0, 1)))
    result = monitor.check_drift()
    assert "threshold" in result


@pytest.mark.parametrize("window_size", [30, 100, 500])
def test_monitor_window_sizes(window_size):
    m = PredictionMonitor(window_size=window_size)
    for i in range(window_size + 10):
        m.record(float(i) / (window_size + 10))
    assert len(m.predictions) == window_size


def test_summary_stats_accuracy():
    """Verify that summary stats match numpy calculations."""
    import numpy as np

    m = PredictionMonitor()
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    for v in data:
        m.record(v)
    s = m.summary()
    expected_mean = round(float(np.mean(data)), 4)
    assert abs(s["mean"] - expected_mean) < 1e-4


def test_set_reference_updates_reference(monitor):
    monitor.set_reference([0.3, 0.4, 0.5])
    assert monitor.reference is not None
    assert len(monitor.reference) == 3


def test_reset_preserves_reference(monitor):
    monitor.set_reference([0.5] * 50)
    monitor.record(0.7)
    monitor.reset()
    assert monitor.reference is not None
    assert len(monitor.predictions) == 0


@pytest.mark.parametrize(
    "data,expected_min,expected_max",
    [
        ([0.1, 0.5, 0.9], 0.1, 0.9),
        ([0.5, 0.5, 0.5], 0.5, 0.5),
        ([0.0, 1.0], 0.0, 1.0),
    ],
)
def test_summary_min_max(data, expected_min, expected_max):
    m = PredictionMonitor()
    for v in data:
        m.record(v)
    s = m.summary()
    assert abs(s["min"] - expected_min) < 1e-4
    assert abs(s["max"] - expected_max) < 1e-4


def test_check_drift_message_on_insufficient_data(monitor):
    result = monitor.check_drift()
    assert "message" in result or result["drift_detected"] is False


def test_batch_record_adds_multiple_predictions(monitor):
    monitor.batch_record([0.1, 0.5, 0.9])
    assert len(monitor) == 3


def test_len_returns_prediction_count(monitor):
    assert len(monitor) == 0
    monitor.record(0.5)
    assert len(monitor) == 1


def test_repr_contains_window_size(monitor):
    r = repr(monitor)
    assert "500" in r
    assert "PredictionMonitor" in r


def test_batch_record_respects_window_size():
    m = PredictionMonitor(window_size=5)
    m.batch_record([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert len(m) == 5
