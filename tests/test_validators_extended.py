"""Extended tests for validators module — medications, criteria, and edge cases."""

from __future__ import annotations

import pytest

from src.validators import (
    is_valid_atc,
    is_valid_icd10,
    validate_criteria_list,
    validate_patient_conditions,
    validate_patient_medications,
)


class TestValidatePatientMedications:
    @pytest.mark.parametrize("code,expected_warnings", [
        ("C09AA01", 0),  # valid ATC
        ("INVALID", 1),  # invalid code
        ("", 0),         # empty code skipped
    ])
    def test_medication_code_validation(self, code, expected_warnings):
        meds = [{"code": code, "name": "Drug"}] if code else [{"name": "Drug"}]
        warnings = validate_patient_medications(meds)
        assert len(warnings) == expected_warnings

    def test_multiple_medications_all_valid(self):
        meds = [
            {"code": "C09AA01", "name": "Lisinopril"},
            {"code": "A10BA02", "name": "Metformin"},
        ]
        assert validate_patient_medications(meds) == []

    def test_multiple_medications_with_invalid(self):
        meds = [{"code": "BAD_CODE", "name": "X"}, {"code": "C09AA01", "name": "Y"}]
        warnings = validate_patient_medications(meds)
        assert len(warnings) == 1
        assert "BAD_CODE" in warnings[0]

    def test_empty_medications_list(self):
        assert validate_patient_medications([]) == []

    def test_string_medication_entry(self):
        warnings = validate_patient_medications(["NOTVALID"])
        assert len(warnings) == 1


class TestValidateCriteriaList:
    def test_valid_criterion_no_warnings(self):
        criteria = [{"field": "age", "operator": "GT", "value": 18}]
        assert validate_criteria_list(criteria) == []

    def test_missing_field_generates_warning(self):
        criteria = [{"operator": "EQ", "value": "male"}]
        warnings = validate_criteria_list(criteria)
        assert any("missing 'field'" in w for w in warnings)

    def test_unknown_operator_generates_warning(self):
        criteria = [{"field": "age", "operator": "BETWEEN", "value": [18, 65]}]
        warnings = validate_criteria_list(criteria)
        assert any("BETWEEN" in w for w in warnings)

    def test_non_dict_entry_generates_warning(self):
        warnings = validate_criteria_list(["not a dict"])
        assert any("not a dict" in w for w in warnings)

    def test_empty_criteria_no_warnings(self):
        assert validate_criteria_list([]) == []

    @pytest.mark.parametrize("operator", ["EQ", "GT", "LT", "GTE", "LTE", "IN", "EXISTS", "NOT_EXISTS"])
    def test_all_valid_operators_accepted(self, operator):
        criteria = [{"field": "some_field", "operator": operator, "value": None}]
        assert validate_criteria_list(criteria) == []


class TestValidatePatientConditionsExtended:
    def test_condition_without_code_key(self):
        conditions = [{"name": "No code here"}]
        warnings = validate_patient_conditions(conditions)
        assert warnings == []

    def test_icd10_with_alphanumeric_suffix_accepted(self):
        assert is_valid_icd10("Z00.00A")

    def test_atc_exact_7_chars_accepted(self):
        assert is_valid_atc("C09AA01")

    def test_atc_wrong_length_rejected(self):
        assert not is_valid_atc("C09AA")

    def test_icd10_lowercase_rejected(self):
        assert not is_valid_icd10("i10")
