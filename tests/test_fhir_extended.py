"""Extended FHIR client tests."""

import pytest

from src.fhir import FHIRClient


@pytest.fixture
def fhir():
    return FHIRClient(base_url="http://localhost:8080/fhir", timeout=1.0)


def test_parse_condition_extracts_code(fhir):
    fhir_cond = {
        "resourceType": "Condition",
        "code": {
            "coding": [{"code": "I10", "display": "Hypertension"}],
            "text": "Hypertension",
        },
    }
    parsed = fhir.parse_condition(fhir_cond)
    assert parsed["icd10_code"] == "I10"
    assert parsed["display"] == "Hypertension"


def test_parse_medication_extracts_code(fhir):
    fhir_med = {
        "resourceType": "MedicationStatement",
        "status": "active",
        "medicationCodeableConcept": {
            "coding": [{"code": "B01AA03", "display": "Warfarin"}],
            "text": "Warfarin",
        },
    }
    parsed = fhir.parse_medication(fhir_med)
    assert parsed["medication_code"] == "B01AA03"
    assert parsed["status"] == "active"


def test_parse_telecom_helper(fhir):
    telecom = [
        {"system": "email", "value": "test@example.com"},
        {"system": "phone", "value": "555-0100"},
    ]
    result = fhir._parse_telecom(telecom)
    assert result["email"] == "test@example.com"
    assert result["phone"] == "555-0100"


def test_parse_telecom_empty(fhir):
    result = fhir._parse_telecom([])
    assert result == {}


def test_mock_patient_has_expected_structure(fhir):
    mock = fhir._mock_patient("pt-test")
    assert mock["resourceType"] == "Patient"
    assert mock["id"] == "pt-test"
    assert "name" in mock
    assert "birthDate" in mock


def test_mock_conditions_returns_list(fhir):
    conds = fhir._mock_conditions()
    assert isinstance(conds, list)
    assert len(conds) > 0
    assert conds[0]["resourceType"] == "Condition"


def test_mock_medications_returns_list(fhir):
    meds = fhir._mock_medications()
    assert isinstance(meds, list)
    assert len(meds) > 0
    assert meds[0]["resourceType"] == "MedicationStatement"


def test_base_url_strips_trailing_slash():
    client = FHIRClient(base_url="http://example.com/fhir/")
    assert not client.base_url.endswith("/")


@pytest.mark.parametrize("patient_id", ["pt-001", "pt-abc-xyz", "12345"])
def test_mock_patient_id_matches(fhir, patient_id):
    mock = fhir._mock_patient(patient_id)
    assert mock["id"] == patient_id


def test_get_patient_falls_back_to_mock_on_connection_error(fhir):
    """get_patient should return mock data when FHIR server is unreachable."""
    result = fhir.get_patient("pt-unreachable")
    assert result["resourceType"] == "Patient"
    assert "name" in result


def test_get_patient_conditions_falls_back_to_mock(fhir):
    """get_patient_conditions returns mock list when server unreachable."""
    result = fhir.get_patient_conditions("pt-unreachable")
    assert isinstance(result, list)
    assert all(r["resourceType"] == "Condition" for r in result)


def test_get_patient_medications_falls_back_to_mock(fhir):
    """get_patient_medications returns mock list when server unreachable."""
    result = fhir.get_patient_medications("pt-unreachable")
    assert isinstance(result, list)
    assert all(r["resourceType"] == "MedicationStatement" for r in result)


def test_fetch_complete_profile_falls_back_gracefully(fhir):
    """fetch_complete_patient_profile assembles a full profile from fallback data."""
    profile = fhir.fetch_complete_patient_profile("pt-fallback")
    assert "fhir_id" in profile
    assert isinstance(profile["conditions"], list)
    assert isinstance(profile["medications"], list)


def test_parse_patient_missing_optional_fields(fhir):
    """parse_patient should handle patients with minimal FHIR data."""
    sparse_patient = {
        "resourceType": "Patient",
        "id": "pt-sparse",
        "name": [{"given": ["Jane"], "family": "Doe"}],
        "birthDate": "1990-01-01",
        "gender": "female",
    }
    parsed = fhir.parse_patient(sparse_patient)
    assert parsed["first_name"] == "Jane"
    assert parsed["email"] is None
    assert parsed["phone"] is None


def test_parse_condition_missing_coding(fhir):
    """parse_condition should handle conditions with empty coding list."""
    fhir_cond = {
        "resourceType": "Condition",
        "code": {"coding": [], "text": "Unknown"},
    }
    parsed = fhir.parse_condition(fhir_cond)
    assert parsed["icd10_code"] is None
    assert parsed["display"] == "Unknown"
