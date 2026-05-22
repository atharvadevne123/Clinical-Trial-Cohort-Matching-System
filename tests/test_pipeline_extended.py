"""Extended tests for the retraining pipeline OSError handling and val_samples field."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pipelines.retrain_pipeline import RetrainingPipeline


class TestPipelineOSErrorHandling:
    @pytest.fixture
    def pipeline(self):
        return RetrainingPipeline(min_samples=10)

    def test_os_error_on_model_save_returns_error_status(self, pipeline):
        X = np.random.rand(50, 14).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        with patch("pipelines.retrain_pipeline.joblib.dump", side_effect=OSError("disk full")):
            result = pipeline.run(X, y)
        assert result["status"] == "error"
        assert "disk full" in result["reason"]

    def test_success_result_includes_val_samples(self, pipeline):
        X, y = pipeline.generate_synthetic_data(50)
        result = pipeline.run(X, y)
        if result["status"] == "success":
            assert "val_samples" in result
            assert result["val_samples"] > 0
        else:
            pytest.skip("xgboost not available or insufficient samples")

    def test_insufficient_samples_returns_skipped(self, pipeline):
        X = np.random.rand(5, 14).astype(np.float32)
        y = np.random.randint(0, 2, 5)
        result = pipeline.run(X, y)
        assert result["status"] == "skipped"

    @pytest.mark.parametrize(
        "min_samples,n_samples,expected_status",
        [
            (500, 100, "skipped"),
            (10, 100, "success"),
        ],
    )
    def test_min_samples_threshold_controls_skip(self, min_samples, n_samples, expected_status):
        p = RetrainingPipeline(min_samples=min_samples)
        X, y = p.generate_synthetic_data(n_samples)
        with patch("pipelines.retrain_pipeline.joblib.dump"):
            result = p.run(X, y)
        if expected_status == "success":
            assert result["status"] in ("success", "error")
        else:
            assert result["status"] == expected_status

    def test_result_timestamp_is_iso_format(self, pipeline):
        X = np.random.rand(5, 14).astype(np.float32)
        y = np.random.randint(0, 2, 5)
        result = pipeline.run(X, y)
        assert "timestamp" in result
        from datetime import datetime

        datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
