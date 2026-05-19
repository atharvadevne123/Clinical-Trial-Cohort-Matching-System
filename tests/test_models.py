"""Tests for database models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import Base, Patient, PatientTrialMatch, Trial, _now

TEST_URL = "sqlite:///./test_models.db"
engine = create_engine(TEST_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)


@pytest.fixture
def db():
    session = Session()
    yield session
    session.rollback()
    session.close()


def test_create_patient(db):
    p = Patient(
        id="MOD_P001",
        first_name="Test",
        last_name="User",
        gender="male",
        conditions=[],
        medications=[],
        allergies=[],
    )
    db.add(p)
    db.commit()
    fetched = db.query(Patient).filter(Patient.id == "MOD_P001").first()
    assert fetched is not None
    assert fetched.first_name == "Test"


def test_create_trial(db):
    t = Trial(
        id="MOD_T001",
        name="Model Test Trial",
        phase="Phase 2",
        primary_condition="Diabetes",
        target_enrollment=50,
    )
    db.add(t)
    db.commit()
    fetched = db.query(Trial).filter(Trial.id == "MOD_T001").first()
    assert fetched is not None
    assert fetched.status == "RECRUITING"


def test_patient_trial_match_unique_constraint(db):
    p = Patient(id="MOD_P002", first_name="A", last_name="B", gender="female",
                conditions=[], medications=[], allergies=[])
    t = Trial(id="MOD_T002", name="T", phase="Phase 1",
               primary_condition="Cancer", target_enrollment=10)
    db.add_all([p, t])
    db.commit()

    m1 = PatientTrialMatch(patient_id="MOD_P002", trial_id="MOD_T002",
                           rule_match_score=80.0, ml_match_score=70.0,
                           enrollment_probability=0.7, combined_score=75.0)
    db.add(m1)
    db.commit()

    from sqlalchemy.exc import IntegrityError
    m2 = PatientTrialMatch(patient_id="MOD_P002", trial_id="MOD_T002",
                           rule_match_score=60.0, ml_match_score=50.0,
                           enrollment_probability=0.5, combined_score=55.0)
    db.add(m2)
    with pytest.raises(IntegrityError):
        db.commit()


def test_now_returns_utc():
    ts = _now()
    assert ts.tzinfo is not None


def test_patient_defaults(db):
    p = Patient(id="MOD_P003", first_name="C", last_name="D", gender="other",
                conditions=None, medications=None, allergies=None)
    db.add(p)
    db.commit()
    fetched = db.query(Patient).filter(Patient.id == "MOD_P003").first()
    assert fetched is not None


def test_patient_trial_match_enrolled_defaults_false(db):
    p = Patient(id="MOD_P004", first_name="E", last_name="F", gender="male",
                conditions=[], medications=[], allergies=[])
    t = Trial(id="MOD_T004", name="Default Enroll Trial", phase="Phase 1",
              primary_condition="Diabetes", target_enrollment=20)
    db.add_all([p, t])
    db.commit()
    m = PatientTrialMatch(patient_id="MOD_P004", trial_id="MOD_T004",
                          rule_match_score=60.0, ml_match_score=55.0,
                          enrollment_probability=0.55, combined_score=57.5)
    db.add(m)
    db.commit()
    fetched = db.query(PatientTrialMatch).filter(
        PatientTrialMatch.patient_id == "MOD_P004"
    ).first()
    assert fetched.enrolled is False


def test_trial_default_status_is_recruiting(db):
    t = Trial(id="MOD_T005", name="Status Trial", phase="Phase 3",
              primary_condition="Cancer", target_enrollment=100)
    db.add(t)
    db.commit()
    fetched = db.query(Trial).filter(Trial.id == "MOD_T005").first()
    assert fetched.status == "RECRUITING"


def test_patient_created_at_set_automatically(db):
    p = Patient(id="MOD_P005", first_name="G", last_name="H", gender="female",
                conditions=[], medications=[], allergies=[])
    db.add(p)
    db.commit()
    fetched = db.query(Patient).filter(Patient.id == "MOD_P005").first()
    assert fetched.created_at is not None


@pytest.mark.parametrize("gender", ["male", "female", "other", "unknown"])
def test_patient_gender_values_persist(db, gender):
    pid = f"MOD_GENDER_{gender}"
    p = Patient(id=pid, first_name="X", last_name="Y", gender=gender,
                conditions=[], medications=[], allergies=[])
    db.add(p)
    db.commit()
    fetched = db.query(Patient).filter(Patient.id == pid).first()
    assert fetched.gender == gender
