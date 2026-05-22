"""Extended NLP processor tests."""

import pytest

from src.nlp import ClinicalNLPProcessor


@pytest.fixture
def nlp():
    return ClinicalNLPProcessor()


@pytest.mark.parametrize(
    "neg_phrase,condition",
    [
        ("no hypertension", "hypertension"),
        ("not diabetes", "diabetes"),
        ("without asthma", "asthma"),
        ("denies heart failure", "heart failure"),
        ("negative for cancer", "cancer"),
        ("rules out stroke", "stroke"),
    ],
)
def test_various_negation_phrases(nlp, neg_phrase, condition):
    entities = nlp.extract_entities(f"Patient: {neg_phrase}.")
    condition_texts = [e["text"] for e in entities["conditions"]]
    assert condition not in condition_texts, f"Expected {condition} to be negated in: {neg_phrase}"


def test_multiple_conditions_same_note(nlp):
    note = "Patient has hypertension, diabetes mellitus, and heart failure."
    entities = nlp.extract_entities(note)
    assert len(entities["conditions"]) >= 3


def test_multiple_medications_same_note(nlp):
    note = "Patient takes warfarin, metformin, and lisinopril daily."
    entities = nlp.extract_entities(note)
    med_names = [m["text"] for m in entities["medications"]]
    assert "warfarin" in med_names
    assert "metformin" in med_names
    assert "lisinopril" in med_names


@pytest.mark.parametrize(
    "symptom,canonical",
    [
        ("chest pain", "chest discomfort"),
        ("fatigue", "fatigue"),
        ("headache", "headache"),
        ("nausea", "nausea"),
        ("palpitations", "palpitations"),
    ],
)
def test_symptom_extraction_parametrized(nlp, symptom, canonical):
    entities = nlp.extract_entities(f"Patient reports {symptom}.")
    symptom_canonicals = [s["canonical"] for s in entities["symptoms"]]
    assert canonical in symptom_canonicals


def test_condition_has_icd10_code(nlp):
    entities = nlp.extract_entities("Patient has diabetes mellitus.")
    conditions = {e["text"]: e for e in entities["conditions"]}
    assert "E11" in conditions["diabetes mellitus"]["codes"]


def test_summarize_returns_all_keys(nlp):
    profile = nlp.summarize_clinical_profile("Patient has hypertension.")
    expected_keys = {
        "num_conditions",
        "num_medications",
        "num_symptoms",
        "conditions",
        "medications",
        "symptoms",
        "severity",
        "disease_burden",
    }
    assert expected_keys.issubset(profile.keys())


def test_is_negated_returns_false_for_positive_mention(nlp):
    text = "patient has hypertension"
    assert not nlp._is_negated(text, "hypertension")


def test_is_negated_returns_true_for_negated_mention(nlp):
    text = "patient denies hypertension entirely"
    assert nlp._is_negated(text, "hypertension")


def test_negation_window_does_not_false_positive_on_distant_negation(nlp):
    """Negation word more than 60 chars before keyword should NOT negate it."""
    from src.nlp import _NEGATION_WINDOW_CHARS

    padding = "x" * (_NEGATION_WINDOW_CHARS + 10)
    text = f"no {padding} hypertension"
    assert not nlp._is_negated(text.lower(), "hypertension")


def test_negation_window_constant_is_positive():
    from src.nlp import _NEGATION_WINDOW_CHARS, _SEVERITY_WINDOW_CHARS

    assert _NEGATION_WINDOW_CHARS > 0
    assert _SEVERITY_WINDOW_CHARS > _NEGATION_WINDOW_CHARS


def test_severity_extraction_for_severe_condition(nlp):
    note = "severe diabetes requiring insulin"
    entities = nlp.extract_entities(note)
    assert (
        entities["severity"].get("diabetes mellitus") == "severe"
        or entities["severity"].get("diabetes") == "severe"
    )


def test_empty_text_returns_empty_entities(nlp):
    result = nlp.extract_entities("")
    assert result == {"conditions": [], "medications": [], "symptoms": [], "severity": {}}


@pytest.mark.parametrize(
    "text,expected_count",
    [
        ("", 0),
        ("Patient is healthy.", 0),
        ("Patient has hypertension and diabetes.", 2),
    ],
)
def test_entity_count_parametrized(nlp, text, expected_count):
    result = nlp.extract_entities(text)
    assert len(result["conditions"]) == expected_count
