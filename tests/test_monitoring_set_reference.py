"""Tests for POST /monitoring/set-reference endpoint and TypedDict types."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.monitoring import DriftResult, SummaryResult


class TestSetReferenceEndpoint:
    def _monitoring_client(self):
        """Create a test client with monitoring router."""
        from fastapi.testclient import TestClient
        from src.main import app
        return TestClient(app)

    def test_set_reference_with_valid_probabilities(self):
        client = self._monitoring_client()
        resp = client.post("/monitoring/set-reference", json={"probabilities": [0.1, 0.5, 0.9]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reference_samples"] == 3
        assert data["status"] == "ok"

    def test_set_reference_missing_probabilities_key_returns_400(self):
        client = self._monitoring_client()
        resp = client.post("/monitoring/set-reference", json={"data": [0.1]})
        assert resp.status_code == 400

    def test_set_reference_non_list_returns_400(self):
        client = self._monitoring_client()
        resp = client.post("/monitoring/set-reference", json={"probabilities": 0.5})
        assert resp.status_code == 400

    def test_set_reference_empty_list_accepted(self):
        client = self._monitoring_client()
        resp = client.post("/monitoring/set-reference", json={"probabilities": []})
        assert resp.status_code == 200
        assert resp.json()["reference_samples"] == 0

    @pytest.mark.parametrize("n_probs", [1, 10, 100])
    def test_set_reference_various_sizes(self, n_probs):
        client = self._monitoring_client()
        probs = [i / n_probs for i in range(n_probs)]
        resp = client.post("/monitoring/set-reference", json={"probabilities": probs})
        assert resp.status_code == 200
        assert resp.json()["reference_samples"] == n_probs


class TestDriftResultTypedDict:
    def test_drift_result_keys(self):
        result: DriftResult = {
            "drift_detected": False,
            "ks_statistic": 0.05,
            "p_value": 0.8,
            "sample_size": 100,
        }
        assert result["drift_detected"] is False
        assert result["ks_statistic"] == 0.05

    def test_summary_result_keys(self):
        result: SummaryResult = {
            "count": 50,
            "mean": 0.6,
            "std": 0.1,
            "min": 0.2,
            "max": 0.9,
        }
        assert result["count"] == 50
        assert result["mean"] == 0.6
