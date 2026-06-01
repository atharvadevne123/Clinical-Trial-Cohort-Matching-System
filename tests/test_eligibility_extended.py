"""Extended eligibility matcher edge-case tests."""

from datetime import datetime, timezone

import pytest

from src.eligibility import EligibilityMatcher


@pytest.fixture
def matcher():
    return EligibilityMatcher()


def test_not_exists_operator_with_absent_field(matcher):
    patient = {"id": "P_NE", "conditions": [], "medications": []}
    trial = {
        "id": "T_NE",
        "inclusion_criteria": [{"field": "condition:E11", "operator": "NOT_EXISTS", "value": None}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True


def test_partial_inclusion_criteria_met_is_ineligible(matcher):
    patient = {
        "id": "P_PART",
        "conditions": [{"code": "I10"}],
        "medications": [],
        "date_of_birth": "1990-01-01T00:00:00",
    }
    trial = {
        "id": "T_PART",
        "inclusion_criteria": [
            {"field": "condition:I10", "operator": "EXISTS", "value": None},
            {"field": "age", "operator": "GT", "value": 80},
        ],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is False


def test_multiple_exclusion_violations_reduces_score(matcher):
    patient = {
        "id": "P_EXC",
        "conditions": [{"code": "I10"}, {"code": "E11"}, {"code": "C50"}],
        "medications": [],
    }
    trial = {
        "id": "T_EXC",
        "inclusion_criteria": [],
        "exclusion_criteria": [
            {"field": "condition:I10", "operator": "EXISTS", "value": None},
            {"field": "condition:E11", "operator": "EXISTS", "value": None},
        ],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is False
    assert len(result["violated_exclusion"]) == 2


def test_medication_code_lookup(matcher):
    patient = {
        "id": "P_MED",
        "conditions": [],
        "medications": [{"medication_code": "B01AA03", "name": "Warfarin"}],
    }
    trial = {
        "id": "T_MED",
        "inclusion_criteria": [
            {"field": "medication:B01AA03", "operator": "EXISTS", "value": None}
        ],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True


def test_calculate_age_with_timezone_aware_dob(matcher):
    patient = {
        "id": "P_TZ",
        "date_of_birth": datetime(1980, 6, 15, tzinfo=timezone.utc),
        "conditions": [],
        "medications": [],
    }
    age = matcher._calculate_age(patient)
    assert 40 < age < 50


def test_calculate_age_returns_none_without_dob(matcher):
    age = matcher._calculate_age({"id": "P_NODOB"})
    assert age is None


def test_nested_path_lookup(matcher):
    patient = {
        "id": "P_NEST",
        "address": {"city": "Boston", "state": "MA"},
        "conditions": [],
        "medications": [],
    }
    trial = {
        "id": "T_NEST",
        "inclusion_criteria": [{"field": "address.city", "operator": "EQ", "value": "Boston"}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True


@pytest.mark.parametrize(
    "conditions,expected_code",
    [
        ([{"code": "I10"}], "I10"),
        ([{"icd10_code": "I10"}], "I10"),
        (["I10"], "I10"),
        ([], None),
    ],
)
def test_find_condition_code_variants(matcher, conditions, expected_code):
    result = matcher._find_condition_code({"conditions": conditions}, "I10")
    assert result == expected_code


def test_empty_inclusion_criteria_is_eligible(matcher):
    """Patient should be eligible when trial has no inclusion criteria and no exclusions."""
    patient = {"id": "P_EMPTY", "conditions": [], "medications": []}
    trial = {"id": "T_EMPTY", "inclusion_criteria": [], "exclusion_criteria": []}
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True
    assert result["match_score"] == 100.0


def test_nested_path_missing_key_returns_none(matcher):
    patient = {"id": "P_MISS", "address": {"city": "Boston"}}
    result = matcher._nested_get(patient, "address.zip")
    assert result is None


def test_nested_path_non_dict_intermediate_returns_none(matcher):
    patient = {"id": "P_LIST", "address": ["Boston"]}
    result = matcher._nested_get(patient, "address.city")
    assert result is None


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (5, 3, True),
        (3, 5, False),
        (None, 3, False),
        ("text", 3, False),
    ],
)
def test_gt_operator_parametrized(matcher, a, b, expected):
    assert matcher._gt(a, b) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("male,female", None, False),
        ("male", "male,female,other", True),
        (["male"], "male,female", True),
        (["unknown"], "male,female", False),
    ],
)
def test_in_operator_parametrized(matcher, a, b, expected):
    assert matcher._in(a, b) == expected


def test_score_candidates_returns_sorted_list(matcher):
    """score_candidates should return results sorted by match_score descending."""
    patients = [
        {"id": "P_S1", "conditions": [{"code": "I10"}], "medications": []},
        {"id": "P_S2", "conditions": [], "medications": []},
    ]
    trial = {
        "id": "T_SCORE",
        "inclusion_criteria": [{"field": "condition:I10", "operator": "EXISTS", "value": None}],
        "exclusion_criteria": [],
    }
    results = matcher.score_candidates(patients, trial)
    assert isinstance(results, list)
    assert len(results) == 2
    scores = [r["match_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_score_candidates_includes_patient_id(matcher):
    patients = [{"id": "P_SID", "conditions": [], "medications": []}]
    trial = {"id": "T_SID", "inclusion_criteria": [], "exclusion_criteria": []}
    results = matcher.score_candidates(patients, trial)
    assert results[0]["patient_id"] == "P_SID"


def test_score_candidates_empty_list(matcher):
    results = matcher.score_candidates(
        [], {"id": "T_EMPTY", "inclusion_criteria": [], "exclusion_criteria": []}
    )
    assert results == []


@pytest.mark.parametrize(
    "field,patient_data,expected_not_none",
    [
        ("age", {"date_of_birth": "1975-01-01"}, True),
        ("gender", {"gender": "male"}, True),
        ("condition:I10", {"conditions": [{"code": "I10"}]}, True),
        ("medication:C09AA01", {"medications": [{"code": "C09AA01"}]}, True),
        ("address.city", {"address": {"city": "Boston"}}, True),
        ("nonexistent_field", {}, False),
    ],
)
def test_get_patient_field_variations(matcher, field, patient_data, expected_not_none):
    patient = {"id": "P_FIELD", "conditions": [], "medications": [], **patient_data}
    result = matcher._get_patient_field(patient, field)
    if expected_not_none:
        assert result is not None
    else:
        assert result is None


def test_check_match_weak_score_has_weak_match_reason(matcher):
    patient = {"id": "P_WEAK", "conditions": [], "medications": []}
    trial = {
        "id": "T_WEAK",
        "inclusion_criteria": [
            {"field": "condition:I10", "operator": "EXISTS", "value": None},
            {"field": "condition:E11", "operator": "EXISTS", "value": None},
            {"field": "condition:C50", "operator": "EXISTS", "value": None},
        ],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert "Weak match" in result["reasons"]
