"""Tests for the automated model retraining pipeline."""

import numpy as np
import pytest

from pipelines.retrain_pipeline import RetrainingPipeline


@pytest.fixture
def pipeline():
    return RetrainingPipeline(min_samples=10)


def test_run_skips_when_insufficient_samples(pipeline):
    X = np.random.rand(5, 14).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0])
    result = pipeline.run(X, y)
    assert result["status"] == "skipped"
    assert "Insufficient" in result["reason"]


def test_run_returns_skipped_with_empty_data(pipeline):
    X = np.empty((0, 14), dtype=np.float32)
    y = np.array([])
    result = pipeline.run(X, y)
    assert result["status"] == "skipped"


def test_generate_synthetic_data_shape(pipeline):
    X, y = pipeline.generate_synthetic_data(n=100)
    assert X.shape[0] == 100
    assert X.shape[1] == 14
    assert len(y) == 100


def test_generate_synthetic_data_binary_labels(pipeline):
    _, y = pipeline.generate_synthetic_data(n=200)
    assert set(y).issubset({0, 1})


def test_run_success_with_sufficient_data(tmp_path):
    model_path = str(tmp_path / "test_model.joblib")
    p = RetrainingPipeline(model_path=model_path, min_samples=50)
    X, y = p.generate_synthetic_data(n=100)
    result = p.run(X, y)
    assert result["status"] == "success"
    assert "train_auc" in result
    assert 0.0 <= result["train_auc"] <= 1.0


def test_run_success_returns_sample_count(tmp_path):
    model_path = str(tmp_path / "test_model2.joblib")
    p = RetrainingPipeline(model_path=model_path, min_samples=50)
    X, y = p.generate_synthetic_data(n=200)
    result = p.run(X, y)
    assert result["sample_count"] == 200


def test_run_success_model_file_created(tmp_path):
    import os

    model_path = str(tmp_path / "trained_model.joblib")
    p = RetrainingPipeline(model_path=model_path, min_samples=50)
    X, y = p.generate_synthetic_data(n=100)
    p.run(X, y)
    assert os.path.exists(model_path)


def test_pipeline_min_samples_attribute():
    p = RetrainingPipeline(min_samples=200)
    assert p.min_samples == 200


@pytest.mark.parametrize("n_samples", [50, 100, 300])
def test_generate_synthetic_data_various_sizes(pipeline, n_samples):
    X, y = pipeline.generate_synthetic_data(n=n_samples)
    assert len(X) == n_samples
    assert len(y) == n_samples
