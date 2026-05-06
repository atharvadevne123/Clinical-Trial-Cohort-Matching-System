"""Extended FHIR client tests."""
import pytest
from unittest.mock import patch, MagicMock
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
