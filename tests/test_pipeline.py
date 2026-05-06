"""Tests for the retraining pipeline."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pipelines.retrain_pipeline import RetrainingPipeline


@pytest.fixture
def pipeline():
    return RetrainingPipeline(model_path="/tmp/test_retrain.joblib", min_samples=100)


def test_run_skipped_insufficient_samples(pipeline):
    X = np.random.rand(50, 14).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    result = pipeline.run(X, y)
    assert result["status"] == "skipped"
    assert "Insufficient" in result["reason"]


def test_run_success_with_enough_samples(pipeline):
    X, y = pipeline.generate_synthetic_data(500)
    result = pipeline.run(X, y)
    assert result["status"] in ("success", "error")
    if result["status"] == "success":
        assert 0.0 <= result["train_auc"] <= 1.0
        assert result["sample_count"] == 500


def test_generate_synthetic_data_shape(pipeline):
    X, y = pipeline.generate_synthetic_data(200)
    assert X.shape == (200, 14)
    assert y.shape == (200,)


def test_generate_synthetic_data_binary_labels(pipeline):
    X, y = pipeline.generate_synthetic_data(100)
    assert set(y).issubset({0, 1})


def test_result_has_timestamp(pipeline):
    X = np.random.rand(10, 14).astype(np.float32)
    y = np.random.randint(0, 2, 10)
    result = pipeline.run(X, y)
    assert "timestamp" in result


@pytest.mark.parametrize("n_samples", [100, 500, 1000])
def test_synthetic_data_various_sizes(pipeline, n_samples):
    X, y = pipeline.generate_synthetic_data(n_samples)
    assert len(X) == n_samples
    assert len(y) == n_samples
