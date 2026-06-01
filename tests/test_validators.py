"""Tests for data validation utilities."""

from datetime import datetime

import pytest

from src.validators import (
    is_valid_atc,
    is_valid_icd10,
    validate_criteria_list,
    validate_date_range,
    validate_enrollment_probability,
    validate_patient_conditions,
    validate_patient_medications,
)


@pytest.mark.parametrize(
    "code,expected",
    [
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
    ],
)
def test_is_valid_icd10(code, expected):
    assert is_valid_icd10(code) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ("C09AA01", True),
        ("A10BA02", True),
        ("B01AA03", True),
        ("invalid", False),
        ("C09", False),
        ("", False),
    ],
)
def test_is_valid_atc(code, expected):
    assert is_valid_atc(code) == expected


@pytest.mark.parametrize(
    "prob,expected_valid",
    [
        (0.0, True),
        (0.5, True),
        (1.0, True),
        (-0.1, False),
        (1.1, False),
    ],
)
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


@pytest.mark.parametrize(
    "code,expected",
    [
        ("Z00.00", True),
        ("Z00.00A", True),
        ("M54.5", True),
        ("F32", True),
        ("G43.909", True),
        ("z10", False),
        ("I10.X9999999", False),
    ],
)
def test_is_valid_icd10_extended_formats(code, expected):
    """Test ICD-10 validation with extended alphanumeric suffix formats."""
    assert is_valid_icd10(code) == expected


def test_icd10_lru_cache_returns_consistent_results():
    """Verify lru_cache doesn't corrupt results on repeated calls."""
    for _ in range(5):
        assert is_valid_icd10("I10") is True
        assert is_valid_icd10("bad") is False


@pytest.mark.parametrize("prob", [0.0, 0.001, 0.5, 0.999, 1.0])
def test_validate_probability_boundary_values(prob):
    valid, _ = validate_enrollment_probability(prob)
    assert valid is True


def test_validate_conditions_empty_list():
    warnings = validate_patient_conditions([])
    assert warnings == []


def test_validate_conditions_string_code():
    warnings = validate_patient_conditions(["I10", "E11"])
    assert warnings == []


def test_validators_all_exports():
    """All public symbols should be present in __all__."""
    from src.validators import __all__ as exports

    assert "is_valid_icd10" in exports
    assert "is_valid_atc" in exports
    assert "validate_enrollment_probability" in exports


def test_is_valid_icd10_strips_whitespace():
    """Validator should strip leading/trailing whitespace before matching."""
    assert is_valid_icd10("  I10  ") is True


def test_is_valid_atc_strips_whitespace():
    assert is_valid_atc("  C09AA01  ") is True


@pytest.mark.parametrize("invalid_icd10", ["", "10I", "i10", "INVALID", "12345"])
def test_is_valid_icd10_invalid_formats(invalid_icd10):
    assert is_valid_icd10(invalid_icd10) is False


@pytest.mark.parametrize("valid_atc", ["C09AA01", "A10BA02", "B01AA03"])
def test_is_valid_atc_valid_codes(valid_atc):
    assert is_valid_atc(valid_atc) is True


def test_validate_patient_medications_valid():
    meds = [{"code": "C09AA01"}, {"code": "A10BA02"}]
    warnings = validate_patient_medications(meds)
    assert warnings == []


def test_validate_patient_medications_invalid():
    meds = [{"code": "NOTVALID"}]
    warnings = validate_patient_medications(meds)
    assert len(warnings) == 1


def test_validate_patient_medications_empty():
    assert validate_patient_medications([]) == []


def test_validate_patient_medications_string_codes():
    warnings = validate_patient_medications(["C09AA01", "INVALID"])
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "criteria,expected_warning_count",
    [
        ([{"field": "age", "operator": "GT", "value": 18}], 0),
        ([{"operator": "GT", "value": 18}], 1),
        ([{"field": "age", "operator": "BADOP", "value": 18}], 1),
        (["not_a_dict"], 1),
        ([], 0),
    ],
)
def test_validate_criteria_list(criteria, expected_warning_count):
    warnings = validate_criteria_list(criteria)
    assert len(warnings) == expected_warning_count


def test_validate_date_range_equal_dates():
    dt = datetime(2024, 6, 1)
    valid, msg = validate_date_range(dt, dt)
    assert valid is True


def test_validate_date_range_none_end():
    valid, msg = validate_date_range(datetime(2024, 1, 1), None)
    assert valid is True
