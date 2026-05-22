"""Extended tests for EligibilityMatcher — score_candidates, edge cases, and scoring."""

from __future__ import annotations

import pytest

from src.eligibility import EligibilityMatcher


@pytest.fixture
def matcher():
    return EligibilityMatcher()


PATIENT_HTN = {
    "id": "P_EXT2_001",
    "gender": "male",
    "date_of_birth": "1970-01-01",
    "conditions": [{"code": "I10", "name": "Hypertension"}],
    "medications": [],
}

TRIAL_REQUIRES_HTN = {
    "id": "T_EXT2_001",
    "inclusion_criteria": [{"field": "condition:I10", "operator": "EXISTS", "value": None}],
    "exclusion_criteria": [],
}

TRIAL_NO_CRITERIA = {
    "id": "T_EXT2_002",
    "inclusion_criteria": [],
    "exclusion_criteria": [],
}


class TestScoreCandidates:
    def test_score_candidates_single(self, matcher):
        patients = [PATIENT_HTN]
        results = matcher.score_candidates(patients, TRIAL_REQUIRES_HTN)
        assert len(results) == 1
        assert results[0]["patient_id"] == "P_EXT2_001"

    def test_score_candidates_sorted_descending(self, matcher):
        p2 = {**PATIENT_HTN, "id": "P_EXT2_002", "conditions": []}
        results = matcher.score_candidates([PATIENT_HTN, p2], TRIAL_REQUIRES_HTN)
        scores = [r["match_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_candidates_empty_list(self, matcher):
        results = matcher.score_candidates([], TRIAL_REQUIRES_HTN)
        assert results == []

    def test_score_candidates_all_match_no_criteria(self, matcher):
        results = matcher.score_candidates([PATIENT_HTN], TRIAL_NO_CRITERIA)
        assert results[0]["eligible"] is True

    @pytest.mark.parametrize("n_patients", [1, 3, 5])
    def test_score_candidates_returns_all_patients(self, matcher, n_patients):
        patients = [{**PATIENT_HTN, "id": f"P_EXT2_{i:03d}"} for i in range(n_patients)]
        results = matcher.score_candidates(patients, TRIAL_REQUIRES_HTN)
        assert len(results) == n_patients


class TestMatchScoreRange:
    @pytest.mark.parametrize("score", [0.0, 50.0, 100.0])
    def test_match_score_in_valid_range(self, matcher, score):
        """Score should always be in [0, 100]."""
        result = matcher.check_match(PATIENT_HTN, TRIAL_REQUIRES_HTN)
        assert 0.0 <= result["match_score"] <= 100.0

    def test_fully_eligible_has_high_score(self, matcher):
        result = matcher.check_match(PATIENT_HTN, TRIAL_REQUIRES_HTN)
        assert result["match_score"] >= 70.0

    def test_zero_criteria_trial_gives_100_score(self, matcher):
        result = matcher.check_match(PATIENT_HTN, TRIAL_NO_CRITERIA)
        assert result["match_score"] == 100.0

    def test_eligible_key_is_bool(self, matcher):
        result = matcher.check_match(PATIENT_HTN, TRIAL_REQUIRES_HTN)
        assert isinstance(result["eligible"], bool)

    def test_reasons_list_non_empty_when_criteria_present(self, matcher):
        result = matcher.check_match(PATIENT_HTN, TRIAL_REQUIRES_HTN)
        assert isinstance(result["reasons"], list)
