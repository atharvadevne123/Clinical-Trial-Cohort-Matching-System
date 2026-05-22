"""Data validation utilities for patient and trial record inputs.

Provides validators for common clinical data formats including ICD-10 codes,
date ranges, and enrollment probability bounds.
"""

import logging
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ICD10_PATTERN = re.compile(r"^[A-Z][0-9]{2}(\.[A-Z0-9]{1,5})?$")
_ATC_PATTERN = re.compile(r"^[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}$")


@lru_cache(maxsize=512)
def is_valid_icd10(code: str) -> bool:
    """Check if a string matches the ICD-10 code format.

    Args:
        code: Candidate ICD-10 code string.

    Returns:
        True if the code matches the ICD-10 pattern (e.g. I10, E11.9).
    """
    return bool(_ICD10_PATTERN.match(code.strip()))


@lru_cache(maxsize=512)
def is_valid_atc(code: str) -> bool:
    """Check if a string matches the ATC medication code format.

    Args:
        code: Candidate ATC code string.

    Returns:
        True if the code matches the 7-character ATC format.
    """
    return bool(_ATC_PATTERN.match(code.strip()))


def validate_enrollment_probability(prob: float) -> Tuple[bool, Optional[str]]:
    """Validate that an enrollment probability is within [0.0, 1.0].

    Args:
        prob: Enrollment probability value.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not isinstance(prob, (int, float)):
        return False, f"Probability must be numeric, got {type(prob).__name__}"
    if prob < 0.0 or prob > 1.0:
        return False, f"Probability must be in [0.0, 1.0], got {prob}"
    return True, None


def validate_date_range(
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[bool, Optional[str]]:
    """Validate that start date precedes end date.

    Args:
        start: Start datetime (or None).
        end: End datetime (or None).

    Returns:
        Tuple of (is_valid, error_message). Always valid if either date is None.
    """
    if start is None or end is None:
        return True, None
    if start > end:
        return False, f"start_date {start.date()} must not be after completion_date {end.date()}"
    return True, None


def validate_patient_conditions(conditions: List[Any]) -> List[str]:
    """Return a list of warning messages for conditions with invalid ICD-10 codes.

    Args:
        conditions: List of condition dicts (each with a ``code`` key) or code strings.

    Returns:
        List of warning strings; empty if all codes are valid or absent.
    """
    warnings: List[str] = []
    for cond in conditions:
        code = cond.get("code") or cond.get("icd10_code") if isinstance(cond, dict) else str(cond)
        if code and not is_valid_icd10(str(code)):
            warnings.append(f"Non-standard ICD-10 code: {code!r}")
    return warnings


def validate_patient_medications(medications: List[Any]) -> List[str]:
    """Return a list of warning messages for medications with invalid ATC codes.

    Args:
        medications: List of medication dicts (each with a ``code`` key) or code strings.

    Returns:
        List of warning strings; empty if all codes are valid or absent.
    """
    warnings: List[str] = []
    for med in medications:
        code = med.get("code") or med.get("atc_code") if isinstance(med, dict) else str(med)
        if code and not is_valid_atc(str(code)):
            warnings.append(f"Non-standard ATC code: {code!r}")
    return warnings


def validate_criteria_list(criteria: List[Any]) -> List[str]:
    """Return warning messages for trial criteria entries missing required fields.

    Args:
        criteria: List of criterion dicts; each should have ``field`` and ``operator``.

    Returns:
        List of warning strings; empty if all entries are well-formed.
    """
    warnings: List[str] = []
    valid_operators = {"EQ", "GT", "LT", "GTE", "LTE", "IN", "EXISTS", "NOT_EXISTS"}
    for i, crit in enumerate(criteria):
        if not isinstance(crit, dict):
            warnings.append(f"Criterion[{i}] is not a dict")
            continue
        if not crit.get("field"):
            warnings.append(f"Criterion[{i}] missing 'field'")
        op = str(crit.get("operator", "")).upper()
        if op and op not in valid_operators:
            warnings.append(f"Criterion[{i}] unknown operator {op!r}")
    return warnings


__all__ = [
    "is_valid_icd10",
    "is_valid_atc",
    "validate_enrollment_probability",
    "validate_date_range",
    "validate_patient_conditions",
    "validate_patient_medications",
    "validate_criteria_list",
]
