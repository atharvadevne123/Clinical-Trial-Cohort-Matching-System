"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app, get_db
from src.models import Base

TEST_DB = "sqlite:///./test_api.db"
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


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()


def test_health_returns_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_status_returns_counts():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "patients" in data
    assert "trials" in data
    assert "matches" in data


def test_create_patient():
    payload = {
        "id": "API_P001",
        "first_name": "Carol",
        "last_name": "White",
        "date_of_birth": "1985-07-22T00:00:00",
        "gender": "female",
    }
    response = client.post("/patients", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "API_P001"
    assert data["first_name"] == "Carol"


def test_create_patient_duplicate_returns_409():
    payload = {
        "id": "API_P_DUP",
        "first_name": "Dup",
        "last_name": "User",
        "date_of_birth": "1980-01-01T00:00:00",
        "gender": "male",
    }
    client.post("/patients", json=payload)
    response = client.post("/patients", json=payload)
    assert response.status_code == 409


def test_get_patient_not_found():
    response = client.get("/patients/NONEXISTENT_ID")
    assert response.status_code == 404


def test_list_patients_returns_list():
    response = client.get("/patients?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_create_trial():
    payload = {
        "id": "API_T001",
        "name": "API Test Trial",
        "phase": "Phase 2",
        "primary_condition": "Hypertension",
        "target_enrollment": 100,
    }
    response = client.post("/trials", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "API_T001"


def test_create_trial_duplicate_returns_409():
    payload = {
        "id": "API_T_DUP",
        "name": "Dup Trial",
        "phase": "Phase 1",
        "primary_condition": "Cancer",
        "target_enrollment": 50,
    }
    client.post("/trials", json=payload)
    response = client.post("/trials", json=payload)
    assert response.status_code == 409


def test_get_trial_not_found():
    response = client.get("/trials/NONEXISTENT_TRIAL")
    assert response.status_code == 404


def test_list_trials_returns_list():
    response = client.get("/trials?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_nlp_extract_entities():
    payload = {"text": "Patient has hypertension and is taking metformin."}
    response = client.post("/nlp/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "extraction_result" in data
    assert data["entities_found"] >= 0


def test_nlp_empty_text_returns_4xx():
    # Pydantic min_length=1 returns 422; empty str after strip returns 400
    payload = {"text": ""}
    response = client.post("/nlp/extract-entities", json=payload)
    assert response.status_code in (400, 422)


def test_nlp_clinical_profile():
    payload = {"text": "Patient has type 2 diabetes, hypertension, and heart failure."}
    response = client.post("/nlp/clinical-profile", json=payload)
    assert response.status_code == 200
    assert "clinical_profile" in response.json()


def test_ml_model_info():
    response = client.get("/ml/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) == 14


@pytest.mark.parametrize("endpoint", ["/", "/health", "/status"])
def test_meta_endpoints_return_200(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 200
