"""Tests for PatientUpdate and TrialUpdate partial update schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import PatientUpdate, TrialUpdate


class TestPatientUpdateSchema:
    def test_all_fields_optional(self):
        update = PatientUpdate()
        assert update.first_name is None
        assert update.gender is None
        assert update.conditions is None

    def test_partial_update_only_first_name(self):
        update = PatientUpdate(first_name="Alice")
        assert update.first_name == "Alice"
        assert update.last_name is None

    def test_model_dump_exclude_none(self):
        update = PatientUpdate(first_name="Bob", gender="male")
        data = update.model_dump(exclude_none=True)
        assert set(data.keys()) == {"first_name", "gender"}

    def test_email_field_accepted(self):
        update = PatientUpdate(email="test@example.com")
        assert update.email == "test@example.com"

    def test_empty_conditions_list_accepted(self):
        update = PatientUpdate(conditions=[])
        assert update.conditions == []

    def test_conditions_with_entries_accepted(self):
        update = PatientUpdate(conditions=[{"code": "I10", "name": "HTN"}])
        assert len(update.conditions) == 1

    @pytest.mark.parametrize(
        "name,length",
        [
            ("first_name", 1),
            ("last_name", 1),
        ],
    )
    def test_min_length_enforced(self, name, length):
        with pytest.raises(ValidationError):
            PatientUpdate(**{name: ""})

    def test_gender_field_accepted_as_string(self):
        update = PatientUpdate(gender="female")
        assert update.gender == "female"


class TestTrialUpdateSchema:
    def test_all_fields_optional(self):
        update = TrialUpdate()
        assert update.name is None
        assert update.phase is None
        assert update.target_enrollment is None

    def test_partial_update_name_only(self):
        update = TrialUpdate(name="New Name")
        data = update.model_dump(exclude_none=True)
        assert data == {"name": "New Name"}

    def test_target_enrollment_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrialUpdate(target_enrollment=0)

    def test_inclusion_criteria_list_accepted(self):
        update = TrialUpdate(inclusion_criteria=[{"field": "age", "operator": "GT", "value": 18}])
        assert len(update.inclusion_criteria) == 1

    def test_model_dump_exclude_none_returns_only_set_fields(self):
        update = TrialUpdate(phase="Phase 3", target_enrollment=100)
        data = update.model_dump(exclude_none=True)
        assert set(data.keys()) == {"phase", "target_enrollment"}

    @pytest.mark.parametrize("phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
    def test_valid_phases_accepted_as_strings(self, phase):
        update = TrialUpdate(phase=phase)
        assert update.phase == phase
