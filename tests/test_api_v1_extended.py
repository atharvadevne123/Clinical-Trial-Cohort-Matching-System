"""Tests for the versioned API v1 router."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app, get_db
from src.models import Base

TEST_DB = "sqlite:///./test_api_v1_ext.db"
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


@pytest.mark.parametrize("endpoint", ["/api/v1/health", "/api/v1/status"])
def test_v1_meta_endpoints_return_200(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 200


def test_v1_health_returns_version():
    response = client.get("/api/v1/health")
    data = response.json()
    assert data.get("version") == "v1"
    assert data.get("status") == "healthy"


def test_v1_status_includes_api_version():
    response = client.get("/api/v1/status")
    data = response.json()
    assert data.get("api_version") == "v1"
    assert "patients" in data
    assert "trials" in data
    assert "matches" in data


def test_v1_list_patients_returns_list():
    response = client.get("/api/v1/patients")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_v1_list_trials_returns_list():
    response = client.get("/api/v1/trials")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_v1_get_patient_not_found():
    response = client.get("/api/v1/patients/NONEXISTENT_V1")
    assert response.status_code == 404


def test_v1_get_trial_not_found():
    response = client.get("/api/v1/trials/NONEXISTENT_V1")
    assert response.status_code == 404


def test_v1_patients_pagination():
    response = client.get("/api/v1/patients?skip=0&limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) <= 5


def test_v1_trials_pagination():
    response = client.get("/api/v1/trials?skip=0&limit=5")
    assert response.status_code == 200
    assert len(response.json()) <= 5


def test_v1_patients_limit_validation():
    response = client.get("/api/v1/patients?limit=0")
    assert response.status_code == 422


def test_v1_operators_endpoint():
    response = client.get("/api/v1/operators")
    assert response.status_code == 200
    data = response.json()
    assert "operators" in data
    assert data.get("api_version") == "v1"
    assert isinstance(data["operators"], list)
    assert len(data["operators"]) > 0


def test_v1_version_endpoint():
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data.get("api_version") == "v1"
