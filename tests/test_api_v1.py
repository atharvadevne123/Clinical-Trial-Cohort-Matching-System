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
