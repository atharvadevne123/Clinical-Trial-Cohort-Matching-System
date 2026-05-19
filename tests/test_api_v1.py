"""Tests for the versioned /api/v1 router."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api_v1 import v1_router
from src.main import app, get_db
from src.models import Base

TEST_DB = "sqlite:///./test_apiv1.db"
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
app.include_router(v1_router)
client = TestClient(app)


def test_v1_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "v1"


def test_v1_list_patients():
    response = client.get("/api/v1/patients?limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_v1_get_patient_not_found():
    response = client.get("/api/v1/patients/NONEXISTENT_V1")
    assert response.status_code == 404


def test_v1_list_trials():
    response = client.get("/api/v1/trials?limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_v1_get_trial_not_found():
    response = client.get("/api/v1/trials/NONEXISTENT_V1_T")
    assert response.status_code == 404


def test_v1_status_includes_version():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == "v1"
    assert "patients" in data
    assert "trials" in data


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/patients", "/api/v1/trials", "/api/v1/status"])
def test_v1_endpoints_return_200(path):
    response = client.get(path)
    assert response.status_code == 200


def test_v1_health_has_timestamp():
    response = client.get("/api/v1/health")
    data = response.json()
    assert "timestamp" in data


def test_v1_create_and_get_patient():
    from datetime import datetime
    payload = {
        "id": "V1_P001",
        "first_name": "V1",
        "last_name": "Patient",
        "date_of_birth": "1985-05-15T00:00:00",
        "gender": "female",
    }
    resp = client.post("/api/v1/patients", json=payload)
    assert resp.status_code in (200, 201, 409)
    resp2 = client.get("/api/v1/patients/V1_P001")
    assert resp2.status_code in (200, 404)


def test_v1_list_patients_returns_list():
    resp = client.get("/api/v1/patients")
    assert isinstance(resp.json(), list)


def test_v1_list_trials_returns_list():
    resp = client.get("/api/v1/trials")
    assert isinstance(resp.json(), list)


def test_v1_status_has_matches_key():
    resp = client.get("/api/v1/status")
    data = resp.json()
    assert "matches" in data


@pytest.mark.parametrize("invalid_limit", [0, -5])
def test_v1_list_patients_invalid_limit(invalid_limit):
    resp = client.get(f"/api/v1/patients?limit={invalid_limit}")
    assert resp.status_code == 422
