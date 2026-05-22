"""Tests for FHIR client retry logic and _get_with_retry method."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from src.fhir import FHIRClient


class TestFHIRRetryLogic:
    def test_retry_on_connect_error_then_success(self):
        client = FHIRClient(base_url="http://fhir-test:8080/fhir", timeout=1.0)
        success_response = MagicMock(spec=httpx.Response)
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {"resourceType": "Patient", "id": "P1"}

        with patch("src.fhir.httpx.get") as mock_get, patch("src.fhir.time.sleep"):
            mock_get.side_effect = [
                httpx.ConnectError("refused"),
                success_response,
            ]
            result = client._get_with_retry("http://fhir-test:8080/fhir/Patient/P1")
        assert result.json() == {"resourceType": "Patient", "id": "P1"}

    def test_all_retries_exhausted_raises(self):
        client = FHIRClient(base_url="http://fhir-test:8080/fhir", timeout=1.0)
        with patch("src.fhir.httpx.get", side_effect=httpx.ConnectError("refused")), \
             patch("src.fhir.time.sleep"):
            with pytest.raises(httpx.ConnectError):
                client._get_with_retry("http://fhir-test:8080/fhir/Patient/P1")

    def test_http_status_error_not_retried(self):
        client = FHIRClient(base_url="http://fhir-test:8080/fhir", timeout=1.0)
        bad_response = MagicMock(spec=httpx.Response)
        bad_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        with patch("src.fhir.httpx.get", return_value=bad_response):
            with pytest.raises(httpx.HTTPStatusError):
                client._get_with_retry("http://fhir-test:8080/fhir/Patient/MISSING")

    def test_get_patient_falls_back_on_all_retries_exhausted(self):
        client = FHIRClient(base_url="http://fhir-test:8080/fhir", timeout=1.0)
        with patch.object(client, "_get_with_retry", side_effect=httpx.ConnectError("down")):
            result = client.get_patient("P_FALLBACK")
        assert result["id"] == "P_FALLBACK"

    def test_max_retries_is_three(self):
        assert FHIRClient._MAX_RETRIES == 3

    def test_retry_backoff_is_positive(self):
        assert FHIRClient._RETRY_BACKOFF > 0

    @pytest.mark.parametrize("timeout", [1.0, 5.0, 10.0])
    def test_custom_timeout_stored(self, timeout):
        client = FHIRClient(timeout=timeout)
        assert client.timeout == timeout

    def test_get_patient_conditions_falls_back_on_error(self):
        client = FHIRClient(base_url="http://fhir-test:8080/fhir", timeout=1.0)
        with patch.object(client, "_get_with_retry", side_effect=Exception("boom")):
            result = client.get_patient_conditions("P1")
        assert isinstance(result, list)

    def test_get_patient_medications_falls_back_on_error(self):
        client = FHIRClient(base_url="http://fhir-test:8080/fhir", timeout=1.0)
        with patch.object(client, "_get_with_retry", side_effect=Exception("boom")):
            result = client.get_patient_medications("P1")
        assert isinstance(result, list)
