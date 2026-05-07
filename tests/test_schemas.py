"""Tests for Pydantic schemas."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas import PatientCreate, TrialCreate


def test_patient_create_valid():
    p = PatientCreate(
        id="P001",
        first_name="Alice",
        last_name="Smith",
        date_of_birth=datetime(1980, 1, 15),
        gender="female",
    )
    assert p.id == "P001"
    assert p.conditions == []


def test_patient_create_missing_required_fields():
    with pytest.raises(ValidationError):
        PatientCreate(first_name="Alice")


def test_trial_create_valid():
    t = TrialCreate(
        id="T001",
        name="Hypertension Trial",
        phase="Phase 2",
        primary_condition="Hypertension",
        target_enrollment=50,
    )
    assert t.id == "T001"
    assert t.inclusion_criteria == []


def test_trial_create_missing_required():
    with pytest.raises(ValidationError):
        TrialCreate(id="T002", name="Trial")


@pytest.mark.parametrize("gender", ["male", "female", "other", "unknown"])
def test_patient_create_various_genders(gender):
    p = PatientCreate(
        id="P002",
        first_name="Test",
        last_name="User",
        date_of_birth=datetime(1990, 6, 1),
        gender=gender,
    )
    assert p.gender == gender


def test_patient_optional_fields_default_to_none():
    p = PatientCreate(
        id="P003",
        first_name="Bob",
        last_name="Jones",
        date_of_birth=datetime(1975, 3, 20),
        gender="male",
    )
    assert p.email is None
    assert p.phone is None
    assert p.postal_code is None


def test_trial_optional_dates_default_to_none():
    t = TrialCreate(
        id="T003",
        name="Study",
        phase="Phase 1",
        primary_condition="Cancer",
        target_enrollment=30,
    )
    assert t.start_date is None
    assert t.completion_date is None
