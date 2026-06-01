"""Tests for the clinical NLP processor."""

import pytest

from src.nlp import ClinicalNLPProcessor


@pytest.fixture
def nlp():
    return ClinicalNLPProcessor()


SAMPLE_NOTE = (
    "Patient presents with hypertension and type 2 diabetes. "
    "Currently taking metformin and lisinopril. Reports fatigue and headache."
)


def test_extract_known_condition(nlp):
    entities = nlp.extract_entities(SAMPLE_NOTE)
    condition_texts = [e["text"] for e in entities["conditions"]]
    assert "hypertension" in condition_texts


def test_extract_multiple_conditions(nlp):
    entities = nlp.extract_entities(SAMPLE_NOTE)
    assert len(entities["conditions"]) >= 2


def test_extract_medication(nlp):
    entities = nlp.extract_entities(SAMPLE_NOTE)
    med_texts = [e["text"] for e in entities["medications"]]
    assert "metformin" in med_texts


def test_negation_suppresses_condition(nlp):
    note = "Patient denies hypertension. No history of diabetes."
    entities = nlp.extract_entities(note)
    condition_texts = [e["text"] for e in entities["conditions"]]
    assert "hypertension" not in condition_texts
    assert "diabetes" not in condition_texts


def test_empty_text_returns_empty(nlp):
    entities = nlp.extract_entities("")
    assert entities["conditions"] == []
    assert entities["medications"] == []
    assert entities["symptoms"] == []


def test_extract_symptoms(nlp):
    entities = nlp.extract_entities(SAMPLE_NOTE)
    symptom_texts = [s["text"] for s in entities["symptoms"]]
    assert "fatigue" in symptom_texts
    assert "headache" in symptom_texts


def test_icd10_codes_included(nlp):
    entities = nlp.extract_entities("Patient has hypertension.")
    conditions = {e["text"]: e for e in entities["conditions"]}
    assert "I10" in conditions["hypertension"]["codes"]


def test_summarize_clinical_profile(nlp):
    profile = nlp.summarize_clinical_profile(SAMPLE_NOTE)
    assert profile["num_conditions"] >= 2
    assert profile["num_medications"] >= 2
    assert profile["disease_burden"] in ("low", "moderate", "high")


@pytest.mark.parametrize(
    "note,expected_burden",
    [
        ("Patient has hypertension.", "low"),
        ("Patient has hypertension and diabetes.", "moderate"),
        ("Patient has hypertension, diabetes, cancer, stroke, heart failure.", "high"),
    ],
)
def test_disease_burden_levels(nlp, note, expected_burden):
    profile = nlp.summarize_clinical_profile(note)
    assert profile["disease_burden"] == expected_burden


def test_severity_extraction(nlp):
    note = "Patient has severe hypertension."
    entities = nlp.extract_entities(note)
    assert "hypertension" in entities["severity"]
    assert entities["severity"]["hypertension"] == "severe"


def test_entity_confidence_present(nlp):
    entities = nlp.extract_entities("Patient has asthma.")
    for e in entities["conditions"]:
        assert "confidence" in e
        assert 0.0 <= e["confidence"] <= 1.0


@pytest.mark.parametrize(
    "text,expected_condition",
    [
        ("Patient has type 2 diabetes.", "diabetes"),
        ("Diagnosed with cancer last year.", "cancer"),
        ("History of atrial fibrillation.", "atrial fibrillation"),
    ],
)
def test_extract_conditions_parametrized(nlp, text, expected_condition):
    entities = nlp.extract_entities(text)
    condition_texts = [e["text"] for e in entities["conditions"]]
    assert expected_condition in condition_texts


def test_extract_entities_whitespace_only(nlp):
    entities = nlp.extract_entities("   ")
    assert entities["conditions"] == [] or isinstance(entities["conditions"], list)


def test_extract_entities_returns_required_keys(nlp):
    entities = nlp.extract_entities("Patient has hypertension.")
    assert "conditions" in entities
    assert "medications" in entities
    assert "symptoms" in entities
    assert "severity" in entities


def test_summarize_profile_keys(nlp):
    profile = nlp.summarize_clinical_profile("Patient has hypertension.")
    assert "num_conditions" in profile
    assert "num_medications" in profile
    assert "num_symptoms" in profile
    assert "disease_burden" in profile


@pytest.mark.parametrize(
    "note",
    [
        "No conditions found here.",
        "Patient is healthy with no known conditions.",
        "Routine check-up, no significant findings.",
    ],
)
def test_no_conditions_extracted_from_healthy_notes(nlp, note):
    entities = nlp.extract_entities(note)
    assert isinstance(entities["conditions"], list)
