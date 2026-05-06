"""Tests for the FHIR R4 client."""
import pytest
from unittest.mock import patch, MagicMock
from src.fhir import FHIRClient


@pytest.fixture
def fhir():
    return FHIRClient(base_url="http://localhost:8080/fhir", timeout=1.0)


def test_get_patient_falls_back_to_mock_on_error(fhir):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        result = fhir.get_patient("patient-123")
    assert "id" in result


def test_get_conditions_falls_back_to_mock(fhir):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        result = fhir.get_patient_conditions("patient-123")
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_medications_falls_back_to_mock(fhir):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        result = fhir.get_patient_medications("patient-123")
    assert isinstance(result, list)


def test_parse_patient_extracts_name(fhir):
    fhir_data = {
        "id": "pt-1",
        "name": [{"given": ["John"], "family": "Doe"}],
        "birthDate": "1970-05-15",
        "gender": "male",
        "telecom": [],
        "address": [],
    }
    parsed = fhir.parse_patient(fhir_data)
    assert parsed["first_name"] == "John"
    assert parsed["last_name"] == "Doe"
    assert parsed["gender"] == "male"


def test_parse_patient_extracts_telecom(fhir):
    fhir_data = {
        "id": "pt-2",
        "name": [{"given": ["Jane"], "family": "Doe"}],
        "birthDate": "1980-01-01",
        "gender": "female",
        "telecom": [
            {"system": "email", "value": "jane@example.com"},
            {"system": "phone", "value": "555-1234"},
        ],
        "address": [],
    }
    parsed = fhir.parse_patient(fhir_data)
    assert parsed["email"] == "jane@example.com"
    assert parsed["phone"] == "555-1234"


def test_fetch_complete_patient_profile_returns_dict(fhir):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        profile = fhir.fetch_complete_patient_profile("patient-999")
    assert isinstance(profile, dict)
    assert "conditions" in profile
    assert "medications" in profile


def test_fhir_client_default_url():
    client = FHIRClient()
    assert "fhir" in client.base_url.lower()
