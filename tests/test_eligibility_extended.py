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
        "inclusion_criteria": [{"field": "medication:B01AA03", "operator": "EXISTS", "value": None}],
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


@pytest.mark.parametrize("conditions,expected_code", [
    ([{"code": "I10"}], "I10"),
    ([{"icd10_code": "I10"}], "I10"),
    (["I10"], "I10"),
    ([], None),
])
def test_find_condition_code_variants(matcher, conditions, expected_code):
    result = matcher._find_condition_code({"conditions": conditions}, "I10")
    assert result == expected_code
