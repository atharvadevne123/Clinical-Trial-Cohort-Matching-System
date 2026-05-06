"""Pytest fixtures for Clinical Trial Cohort Matching System tests."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import Base, Patient, Trial
from src.main import app, get_db


@pytest.fixture(scope="session")
def engine():
    eng = create_engine("sqlite:///./test_clinical.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture
def db_session(engine):
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_patient(db_session):
    patient = Patient(
        id="P001",
        first_name="Alice",
        last_name="Smith",
        gender="female",
        conditions=[{"code": "I10", "name": "Hypertension"}],
        medications=[{"code": "C09AA01", "name": "Lisinopril"}],
        allergies=[],
    )
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)
    yield patient
    db_session.delete(patient)
    db_session.commit()


@pytest.fixture
def sample_trial(db_session):
    trial = Trial(
        id="T001",
        name="Hypertension Study",
        phase="Phase 2",
        primary_condition="Hypertension",
        target_enrollment=100,
        inclusion_criteria=[{"field": "condition:I10", "operator": "EXISTS", "value": None}],
        exclusion_criteria=[],
    )
    db_session.add(trial)
    db_session.commit()
    db_session.refresh(trial)
    yield trial
    db_session.delete(trial)
    db_session.commit()
