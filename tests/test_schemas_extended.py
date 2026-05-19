"""Extended schema validation tests."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas import ClinicalNoteRequest, PatientCreate, TrialCreate


def test_patient_gender_normalised_to_lowercase():
    p = PatientCreate(
        id="P_NORM",
        first_name="Test",
        last_name="User",
        date_of_birth=datetime(1990, 1, 1),
        gender="MALE",
    )
    assert p.gender == "male"


def test_patient_invalid_gender_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        PatientCreate(
            id="P_INV",
            first_name="Test",
            last_name="User",
            date_of_birth=datetime(1990, 1, 1),
            gender="cyborg",
        )
    assert "gender" in str(exc_info.value)


def test_patient_invalid_email_raises_validation_error():
    with pytest.raises(ValidationError):
        PatientCreate(
            id="P_EMAIL",
            first_name="Test",
            last_name="User",
            date_of_birth=datetime(1990, 1, 1),
            gender="male",
            email="not-an-email",
        )


def test_patient_valid_email_accepted():
    p = PatientCreate(
        id="P_EMAIL2",
        first_name="Test",
        last_name="User",
        date_of_birth=datetime(1990, 1, 1),
        gender="female",
        email="test@example.com",
    )
    assert p.email == "test@example.com"


def test_trial_invalid_phase_raises_error():
    with pytest.raises(ValidationError):
        TrialCreate(
            id="T_INV",
            name="Invalid Trial",
            phase="Phase 99",
            primary_condition="Cancer",
            target_enrollment=50,
        )


@pytest.mark.parametrize("phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 1/2", "Phase 2/3", "N/A"])
def test_trial_valid_phases(phase):
    t = TrialCreate(
        id=f"T_{phase.replace(' ', '_')}",
        name="Phase Trial",
        phase=phase,
        primary_condition="Diabetes",
        target_enrollment=100,
    )
    assert t.phase == phase


def test_trial_target_enrollment_must_be_positive():
    with pytest.raises(ValidationError):
        TrialCreate(
            id="T_NEG",
            name="Neg Trial",
            phase="Phase 1",
            primary_condition="Cancer",
            target_enrollment=0,
        )


def test_clinical_note_request_requires_text():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ClinicalNoteRequest()


def test_clinical_note_request_valid():
    req = ClinicalNoteRequest(text="Patient has hypertension.")
    assert req.text == "Patient has hypertension."


@pytest.mark.parametrize("invalid_email", [
    "notanemail",
    "missing-at-sign",
    "@nodomain",
    "nolocal@",
    "no-dot@domain",
])
def test_patient_invalid_email_formats(invalid_email):
    """All invalid email formats should raise ValidationError."""
    with pytest.raises(ValidationError):
        PatientCreate(
            id="P_BAD_EMAIL",
            first_name="Test",
            last_name="User",
            date_of_birth=datetime(1990, 1, 1),
            gender="male",
            email=invalid_email,
        )


@pytest.mark.parametrize("valid_email", [
    "test@example.com",
    "user.name@domain.org",
    "user+tag@sub.domain.co.uk",
])
def test_patient_valid_email_formats(valid_email):
    p = PatientCreate(
        id="P_GOOD_EMAIL",
        first_name="Test",
        last_name="User",
        date_of_birth=datetime(1990, 1, 1),
        gender="female",
        email=valid_email,
    )
    assert p.email == valid_email


def test_patient_none_email_accepted():
    """email=None should be accepted."""
    p = PatientCreate(
        id="P_NO_EMAIL",
        first_name="Test",
        last_name="User",
        date_of_birth=datetime(1990, 1, 1),
        gender="other",
    )
    assert p.email is None


@pytest.mark.parametrize("gender", ["male", "female", "other", "unknown"])
def test_patient_all_valid_genders(gender):
    p = PatientCreate(
        id=f"P_{gender}",
        first_name="Test",
        last_name="User",
        date_of_birth=datetime(1990, 1, 1),
        gender=gender,
    )
    assert p.gender == gender
