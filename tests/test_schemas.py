"""Tests for Pydantic schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas import ClinicalNoteRequest, PatientCreate, PatientUpdate, TrialCreate, TrialUpdate


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


def test_patient_id_min_length_validation():
    with pytest.raises(ValidationError):
        PatientCreate(
            id="",
            first_name="Alice",
            last_name="Smith",
            date_of_birth=datetime(1980, 1, 1),
            gender="female",
        )


def test_patient_update_partial():
    u = PatientUpdate(email="new@example.com")
    assert u.email == "new@example.com"
    assert u.phone is None


def test_trial_update_partial():
    u = TrialUpdate(name="Updated Name")
    assert u.name == "Updated Name"


def test_clinical_note_request_valid():
    req = ClinicalNoteRequest(text="Patient has hypertension.")
    assert req.text == "Patient has hypertension."


def test_clinical_note_request_empty_fails():
    with pytest.raises(ValidationError):
        ClinicalNoteRequest(text="")


@pytest.mark.parametrize("enrollment", [1, 50, 100, 1000])
def test_trial_create_various_enrollment_sizes(enrollment):
    t = TrialCreate(
        id=f"T_{enrollment}",
        name="Study",
        phase="Phase 3",
        primary_condition="Hypertension",
        target_enrollment=enrollment,
    )
    assert t.target_enrollment == enrollment
