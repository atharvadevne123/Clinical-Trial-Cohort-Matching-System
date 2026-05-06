"""Tests for data validation utilities."""
import pytest
from datetime import datetime
from src.validators import (
    is_valid_icd10, is_valid_atc, validate_enrollment_probability,
    validate_date_range, validate_patient_conditions,
)


@pytest.mark.parametrize("code,expected", [
    ("I10", True),
    ("E11", True),
    ("E11.9", True),
    ("I48.91", True),
    ("C50", True),
    ("X99", True),
    ("invalid", False),
    ("", False),
    ("123", False),
    ("i10", False),
])
def test_is_valid_icd10(code, expected):
    assert is_valid_icd10(code) == expected


@pytest.mark.parametrize("code,expected", [
    ("C09AA01", True),
    ("A10BA02", True),
    ("B01AA03", True),
    ("invalid", False),
    ("C09", False),
    ("", False),
])
def test_is_valid_atc(code, expected):
    assert is_valid_atc(code) == expected


@pytest.mark.parametrize("prob,expected_valid", [
    (0.0, True),
    (0.5, True),
    (1.0, True),
    (-0.1, False),
    (1.1, False),
])
def test_validate_probability(prob, expected_valid):
    valid, msg = validate_enrollment_probability(prob)
    assert valid == expected_valid
    if not expected_valid:
        assert msg is not None


def test_validate_probability_non_numeric():
    valid, msg = validate_enrollment_probability("high")
    assert not valid
    assert msg is not None


def test_validate_date_range_valid():
    start = datetime(2024, 1, 1)
    end = datetime(2025, 1, 1)
    valid, msg = validate_date_range(start, end)
    assert valid
    assert msg is None


def test_validate_date_range_invalid():
    start = datetime(2025, 6, 1)
    end = datetime(2024, 1, 1)
    valid, msg = validate_date_range(start, end)
    assert not valid
    assert msg is not None


def test_validate_date_range_none_start():
    valid, msg = validate_date_range(None, datetime(2025, 1, 1))
    assert valid


def test_validate_conditions_valid_codes():
    conditions = [{"code": "I10"}, {"code": "E11"}]
    warnings = validate_patient_conditions(conditions)
    assert warnings == []


def test_validate_conditions_invalid_code():
    conditions = [{"code": "BAD_CODE"}]
    warnings = validate_patient_conditions(conditions)
    assert len(warnings) == 1


def test_validate_conditions_mixed():
    conditions = [{"code": "I10"}, {"code": "NOT_VALID"}]
    warnings = validate_patient_conditions(conditions)
    assert len(warnings) == 1
