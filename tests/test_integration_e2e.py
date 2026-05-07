"""Integration tests for the full patient-trial matching pipeline."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app, get_db
from src.models import Base, Patient, Trial

TEST_DB = "sqlite:///./test_integration.db"
engine = create_engine(TEST_DB, connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

PATIENT_ID = "INTG_P001"
TRIAL_ID = "INTG_T001"


@pytest.fixture(scope="module", autouse=True)
def setup_data():
    db = TestSession()
    # Clean up any leftover data
    db.query(Patient).filter(Patient.id == PATIENT_ID).delete()
    db.query(Trial).filter(Trial.id == TRIAL_ID).delete()
    db.commit()

    p = Patient(
        id=PATIENT_ID,
        first_name="Integration",
        last_name="Patient",
        gender="male",
        conditions=[{"code": "I10", "name": "Hypertension"}],
        medications=[{"code": "C09AA01", "name": "Lisinopril"}],
        allergies=[],
    )
    t = Trial(
        id=TRIAL_ID,
        name="Integration Hypertension Trial",
        phase="Phase 2",
        primary_condition="Hypertension",
        target_enrollment=50,
        inclusion_criteria=[{"field": "condition:I10", "operator": "EXISTS", "value": None}],
        exclusion_criteria=[],
    )
    db.add_all([p, t])
    db.commit()
    db.close()
    yield
    db = TestSession()
    db.query(Patient).filter(Patient.id == PATIENT_ID).delete()
    db.query(Trial).filter(Trial.id == TRIAL_ID).delete()
    db.commit()
    db.close()


def test_integration_patient_exists():
    response = client.get(f"/patients/{PATIENT_ID}")
    assert response.status_code == 200
    assert response.json()["id"] == PATIENT_ID


def test_integration_trial_exists():
    response = client.get(f"/trials/{TRIAL_ID}")
    assert response.status_code == 200
    assert response.json()["id"] == TRIAL_ID


def test_integration_nlp_extracts_conditions():
    note = "Patient has hypertension. Currently taking lisinopril."
    response = client.post("/nlp/extract-entities", json={"text": note})
    assert response.status_code == 200
    data = response.json()
    condition_texts = [c["text"] for c in data["extraction_result"]["conditions"]]
    assert "hypertension" in condition_texts


def test_integration_ml_predict():
    payload = {
        "patient_id": PATIENT_ID,
        "trial_id": TRIAL_ID,
        "features": {
            "date_of_birth": "1970-01-01",
            "gender": "male",
            "conditions": [{"code": "I10", "name": "Hypertension"}],
            "medications": [],
        },
    }
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["enrollment_probability"] <= 1.0
    assert data["confidence"] in ("HIGH", "MEDIUM", "LOW")


def test_integration_model_info():
    response = client.get("/ml/model/info")
    assert response.status_code == 200
    assert len(response.json()["features"]) == 14


def test_integration_status_shows_records():
    response = client.get("/status")
    data = response.json()
    assert data["patients"] >= 1
    assert data["trials"] >= 1
