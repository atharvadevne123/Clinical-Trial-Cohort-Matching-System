"""Extended API endpoint tests."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import Base
from src.main import app, get_db

TEST_DB = "sqlite:///./test_api_ext.db"
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


def test_version_endpoint_returns_version():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint_returns_uptime():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


def test_metrics_includes_counts():
    response = client.get("/metrics")
    data = response.json()
    assert "patients" in data
    assert "trials" in data
    assert "total_matches" in data


def test_correlation_id_header_in_response():
    response = client.get("/health")
    assert "x-correlation-id" in response.headers


def test_correlation_id_echoed_when_provided():
    headers = {"X-Correlation-ID": "test-corr-123"}
    response = client.get("/health", headers=headers)
    assert response.headers.get("x-correlation-id") == "test-corr-123"


def test_list_patients_skip_limit():
    response = client.get("/patients?skip=0&limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_trials_skip_limit():
    response = client.get("/trials?skip=0&limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_ml_predict_single():
    payload = {
        "patient_id": "P_ML",
        "trial_id": "T_ML",
        "features": {"age": 55, "gender": "male", "conditions": [], "medications": []},
    }
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "enrollment_probability" in data
    assert 0.0 <= data["enrollment_probability"] <= 1.0


def test_ml_predict_batch():
    payload = {
        "patients": [
            {"id": "P1", "age": 55, "gender": "male", "conditions": [], "medications": []},
            {"id": "P2", "age": 35, "gender": "female", "conditions": [], "medications": []},
        ],
        "trial_id": "T_BATCH",
    }
    response = client.post("/ml/predict/batch", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    probs = [r["enrollment_probability"] for r in results]
    assert probs == sorted(probs, reverse=True)


@pytest.mark.parametrize("invalid_limit", [-1, 0, 501])
def test_list_patients_invalid_limit(invalid_limit):
    response = client.get(f"/patients?limit={invalid_limit}")
    assert response.status_code == 422
