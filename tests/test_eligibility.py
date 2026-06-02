"""Tests for the eligibility matching engine."""

import pytest

from src.eligibility import EligibilityMatcher


@pytest.fixture
def matcher():
    return EligibilityMatcher()


@pytest.fixture
def basic_patient():
    return {
        "id": "P001",
        "date_of_birth": "1970-01-01T00:00:00",
        "gender": "male",
        "conditions": [{"code": "I10", "name": "Hypertension"}],
        "medications": [{"code": "C09AA01", "name": "Lisinopril"}],
    }


@pytest.fixture
def basic_trial():
    return {
        "id": "T001",
        "inclusion_criteria": [{"field": "condition:I10", "operator": "EXISTS", "value": None}],
        "exclusion_criteria": [],
    }


def test_eligible_patient_matches_trial(matcher, basic_patient, basic_trial):
    result = matcher.check_match(basic_patient, basic_trial)
    assert result["eligible"] is True
    assert result["match_score"] > 0


def test_patient_without_condition_ineligible(matcher, basic_trial):
    patient = {"id": "P002", "conditions": [], "medications": []}
    result = matcher.check_match(patient, basic_trial)
    assert result["eligible"] is False


def test_exclusion_criterion_makes_patient_ineligible(matcher, basic_patient):
    trial = {
        "id": "T002",
        "inclusion_criteria": [{"field": "condition:I10", "operator": "EXISTS", "value": None}],
        "exclusion_criteria": [{"field": "condition:I10", "operator": "EXISTS", "value": None}],
    }
    result = matcher.check_match(basic_patient, trial)
    assert result["eligible"] is False
    assert len(result["violated_exclusion"]) > 0


def test_no_criteria_gives_full_score(matcher, basic_patient):
    trial = {"id": "T003", "inclusion_criteria": [], "exclusion_criteria": []}
    result = matcher.check_match(basic_patient, trial)
    assert result["eligible"] is True
    assert result["match_score"] == 100.0


def test_age_criterion_gt(matcher):
    patient = {
        "id": "P003",
        "date_of_birth": "1960-01-01T00:00:00",
        "conditions": [],
        "medications": [],
    }
    trial = {
        "id": "T004",
        "inclusion_criteria": [{"field": "age", "operator": "GT", "value": 50}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True


def test_age_criterion_lt_fails_older_patient(matcher):
    patient = {
        "id": "P004",
        "date_of_birth": "1940-01-01T00:00:00",
        "conditions": [],
        "medications": [],
    }
    trial = {
        "id": "T005",
        "inclusion_criteria": [{"field": "age", "operator": "LT", "value": 50}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is False


@pytest.mark.parametrize(
    "operator,value,dob,expected",
    [
        ("GTE", 18, "2000-01-01T00:00:00", True),
        ("LTE", 80, "1970-01-01T00:00:00", True),
        ("EQ", "male", None, False),
    ],
)
def test_parametrized_age_operators(matcher, operator, value, dob, expected):
    patient = {
        "id": "P_PARAM",
        "date_of_birth": dob,
        "gender": "male",
        "conditions": [],
        "medications": [],
    }
    trial = {
        "id": "T_PARAM",
        "inclusion_criteria": [{"field": "age", "operator": operator, "value": value}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] == expected


def test_in_operator_gender(matcher):
    patient = {"id": "P005", "gender": "female", "conditions": [], "medications": []}
    trial = {
        "id": "T006",
        "inclusion_criteria": [{"field": "gender", "operator": "IN", "value": "male,female"}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True


def test_unknown_operator_returns_false(matcher, basic_patient):
    trial = {
        "id": "T007",
        "inclusion_criteria": [{"field": "age", "operator": "UNKNOWN_OP", "value": 30}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(basic_patient, trial)
    assert result["eligible"] is False


def test_match_score_ranges_between_0_and_100(matcher, basic_patient, basic_trial):
    result = matcher.check_match(basic_patient, basic_trial)
    assert 0 <= result["match_score"] <= 100


def test_reasons_list_populated(matcher, basic_patient, basic_trial):
    result = matcher.check_match(basic_patient, basic_trial)
    assert isinstance(result["reasons"], list)
    assert len(result["reasons"]) > 0


@pytest.mark.parametrize(
    "operator,field,value,patient_extra,expected_eligible",
    [
        ("EQ", "gender", "male", {"gender": "male"}, True),
        ("EQ", "gender", "female", {"gender": "male"}, False),
        ("IN", "gender", "male,female,other", {"gender": "female"}, True),
        ("NOT_IN", "gender", "unknown,other", {"gender": "female"}, True),
        ("NOT_IN", "gender", "male,female", {"gender": "female"}, False),
        ("CONTAINS", "first_name", "ali", {"first_name": "Alice"}, True),
        ("CONTAINS", "first_name", "bob", {"first_name": "Alice"}, False),
    ],
)
def test_parametrized_field_operators(
    matcher, operator, field, value, patient_extra, expected_eligible
):
    patient = {"id": "P_PARAM2", "conditions": [], "medications": [], **patient_extra}
    trial = {
        "id": "T_PARAM2",
        "inclusion_criteria": [{"field": field, "operator": operator, "value": value}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] == expected_eligible


def test_score_candidates_returns_sorted_list(matcher):
    patients = [
        {"id": "P_SC1", "date_of_birth": "1990-01-01", "conditions": [], "medications": []},
        {
            "id": "P_SC2",
            "date_of_birth": "1960-01-01",
            "conditions": [{"code": "I10"}],
            "medications": [],
        },
    ]
    trial = {
        "id": "T_SC",
        "inclusion_criteria": [{"field": "condition:I10", "operator": "EXISTS", "value": None}],
        "exclusion_criteria": [],
    }
    results = matcher.score_candidates(patients, trial)
    assert len(results) == 2
    assert results[0]["match_score"] >= results[1]["match_score"]
    assert all("patient_id" in r for r in results)


def test_nested_field_access(matcher):
    patient = {"id": "P_NEST", "address": {"city": "Boston"}, "conditions": [], "medications": []}
    trial = {
        "id": "T_NEST",
        "inclusion_criteria": [{"field": "address.city", "operator": "EQ", "value": "Boston"}],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True


def test_medication_field_access(matcher):
    patient = {
        "id": "P_MED",
        "conditions": [],
        "medications": [{"code": "C09AA01", "name": "Lisinopril"}],
    }
    trial = {
        "id": "T_MED",
        "inclusion_criteria": [
            {"field": "medication:C09AA01", "operator": "EXISTS", "value": None}
        ],
        "exclusion_criteria": [],
    }
    result = matcher.check_match(patient, trial)
    assert result["eligible"] is True


def test_operator_names_property(matcher):
    names = matcher.operator_names
    assert isinstance(names, list)
    assert names == sorted(names)
    assert "EQ" in names
    assert "EXISTS" in names


def test_repr_contains_operators(matcher):
    r = repr(matcher)
    assert "EligibilityMatcher" in r
    assert "EQ" in r
