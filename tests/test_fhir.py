"""Tests for the FHIR R4 client."""

from unittest.mock import patch

import pytest

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


def test_parse_patient_missing_name_fields(fhir):
    fhir_data = {"id": "pt-3", "name": [], "birthDate": "1975-01-01", "gender": "unknown"}
    parsed = fhir.parse_patient(fhir_data)
    assert "id" in parsed or "first_name" in parsed


@pytest.mark.parametrize("patient_id", ["patient-001", "patient-002", "patient-abc"])
def test_get_patient_mock_returns_dict_for_various_ids(fhir, patient_id):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        result = fhir.get_patient(patient_id)
    assert isinstance(result, dict)
    assert "id" in result


def test_fetch_complete_profile_includes_all_sections(fhir):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        profile = fhir.fetch_complete_patient_profile("patient-test")
    for key in ("conditions", "medications"):
        assert key in profile
        assert isinstance(profile[key], list)


def test_fhir_client_custom_base_url():
    client = FHIRClient(base_url="http://custom-fhir.local/r4")
    assert client.base_url == "http://custom-fhir.local/r4"


def test_get_patient_conditions_mock_structure(fhir):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        conditions = fhir.get_patient_conditions("pt-mock")
    for cond in conditions:
        assert "code" in cond or "name" in cond
