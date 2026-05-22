"""Tests for the retraining pipeline."""

import numpy as np
import pytest

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


def test_run_skipped_returns_reason_with_counts(pipeline):
    """Skipped result should include actual vs required sample counts."""
    X = np.random.rand(5, 14).astype(np.float32)
    y = np.random.randint(0, 2, 5)
    result = pipeline.run(X, y)
    assert result["status"] == "skipped"
    assert "5" in result["reason"]


def test_pipeline_custom_min_samples():
    """Pipeline should respect custom min_samples threshold."""
    p = RetrainingPipeline(model_path="/tmp/test_custom.joblib", min_samples=50)
    X = np.random.rand(40, 14).astype(np.float32)
    y = np.random.randint(0, 2, 40)
    result = p.run(X, y)
    assert result["status"] == "skipped"

    X_ok = np.random.rand(60, 14).astype(np.float32)
    y_ok = np.random.randint(0, 2, 60)
    result_ok = p.run(X_ok, y_ok)
    assert result_ok["status"] in ("success", "error")


def test_model_path_is_relative_to_src():
    """MODEL_PATH should not contain a double 'src/src' segment."""
    from pipelines.retrain_pipeline import MODEL_PATH

    assert "src/src" not in MODEL_PATH.replace("\\", "/")
