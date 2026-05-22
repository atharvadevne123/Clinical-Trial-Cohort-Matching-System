"""Tests for NOT_IN and CONTAINS operators in EligibilityMatcher."""

from __future__ import annotations

import pytest

from src.eligibility import SUPPORTED_OPERATORS, EligibilityMatcher


@pytest.fixture
def matcher():
    return EligibilityMatcher()


PATIENT = {
    "id": "P_OP_001",
    "gender": "female",
    "date_of_birth": "1975-03-10",
    "conditions": [{"code": "I10", "name": "Hypertension"}],
    "medications": [],
    "blood_type": "O+",
    "site": "Boston",
}


class TestNotInOperator:
    def test_not_in_as_exclusion_triggers_for_absent_value(self, matcher):
        # Exclusion: "NOT_IN male,unknown" → returns True when gender NOT in list → violation
        trial = {
            "inclusion_criteria": [],
            "exclusion_criteria": [
                {"field": "gender", "operator": "NOT_IN", "value": "male,unknown"}
            ],
        }
        result = matcher.check_match(PATIENT, trial)
        assert len(result["violated_exclusion"]) == 1

    def test_not_in_as_exclusion_no_violation_when_value_in_list(self, matcher):
        # Exclusion: "NOT_IN female,other" → returns False for gender=female → no violation
        trial = {
            "inclusion_criteria": [],
            "exclusion_criteria": [
                {"field": "gender", "operator": "NOT_IN", "value": "female,other"}
            ],
        }
        result = matcher.check_match(PATIENT, trial)
        assert len(result["violated_exclusion"]) == 0

    def test_not_in_with_list_field(self, matcher):
        assert not matcher._not_in(["enrolled", "active"], "enrolled,inactive")

    def test_not_in_none_value(self, matcher):
        assert matcher._not_in(None, "a,b") is True

    def test_not_in_single_option(self, matcher):
        assert matcher._not_in("male", "female") is True
        assert matcher._not_in("female", "female") is False


class TestContainsOperator:
    def test_contains_matching_substring(self, matcher):
        trial = {
            "inclusion_criteria": [{"field": "site", "operator": "CONTAINS", "value": "bost"}],
            "exclusion_criteria": [],
        }
        result = matcher.check_match(PATIENT, trial)
        assert len(result["matched_inclusion"]) == 1

    def test_contains_case_insensitive(self, matcher):
        assert matcher._contains("Boston", "BOST") is True
        assert matcher._contains("Boston", "boston") is True

    def test_contains_no_match(self, matcher):
        assert matcher._contains("Boston", "Chicago") is False

    def test_contains_none_field(self, matcher):
        assert matcher._contains(None, "anything") is False

    def test_contains_none_value(self, matcher):
        assert matcher._contains("something", None) is False

    def test_contains_empty_string(self, matcher):
        assert matcher._contains("anything", "") is True


class TestBetweenOperator:
    def test_between_within_range(self, matcher):
        assert matcher._between(50, "18,65") is True

    def test_between_at_lower_bound(self, matcher):
        assert matcher._between(18, "18,65") is True

    def test_between_at_upper_bound(self, matcher):
        assert matcher._between(65, "18,65") is True

    def test_between_below_range(self, matcher):
        assert matcher._between(17, "18,65") is False

    def test_between_above_range(self, matcher):
        assert matcher._between(66, "18,65") is False

    def test_between_none_value(self, matcher):
        assert matcher._between(None, "18,65") is False

    def test_between_malformed_range(self, matcher):
        assert matcher._between(40, "not-a-range") is False

    def test_between_as_inclusion_criterion(self, matcher):
        patient = {**PATIENT, "date_of_birth": "1980-01-01"}
        trial = {
            "inclusion_criteria": [{"field": "age", "operator": "BETWEEN", "value": "18,65"}],
            "exclusion_criteria": [],
        }
        result = matcher.check_match(patient, trial)
        assert result["match_score"] > 0


class TestSupportedOperators:
    def test_supported_operators_contains_not_in(self):
        assert "NOT_IN" in SUPPORTED_OPERATORS

    def test_supported_operators_contains_contains(self):
        assert "CONTAINS" in SUPPORTED_OPERATORS

    def test_matcher_has_not_in_key(self, matcher):
        assert "NOT_IN" in matcher.operators

    def test_matcher_has_contains_key(self, matcher):
        assert "CONTAINS" in matcher.operators

    def test_all_supported_operators_registered(self, matcher):
        for op in SUPPORTED_OPERATORS:
            assert op in matcher.operators, f"Operator {op} not registered"

    def test_supported_operators_contains_between(self):
        assert "BETWEEN" in SUPPORTED_OPERATORS

    def test_matcher_has_between_key(self, matcher):
        assert "BETWEEN" in matcher.operators
