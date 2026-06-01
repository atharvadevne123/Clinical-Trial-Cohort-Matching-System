"""Extended API endpoint tests."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app, get_db
from src.models import Base

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
    assert data["version"] == "1.2.0"


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


def test_nlp_extract_empty_text_returns_400():
    response = client.post("/nlp/extract-entities", json={"text": "   "})
    assert response.status_code == 400


def test_nlp_extract_valid_text():
    response = client.post("/nlp/extract-entities", json={"text": "Patient has hypertension."})
    assert response.status_code == 200
    data = response.json()
    assert "text_length" in data
    assert "extraction_result" in data


def test_nlp_clinical_profile_empty_text_returns_400():
    response = client.post("/nlp/clinical-profile", json={"text": ""})
    assert response.status_code in (400, 422)


def test_nlp_clinical_profile_valid_text():
    response = client.post("/nlp/clinical-profile", json={"text": "Patient has diabetes."})
    assert response.status_code == 200
    data = response.json()
    assert "clinical_profile" in data


def test_get_nonexistent_patient_returns_404():
    response = client.get("/patients/NO_SUCH_PATIENT_XYZ")
    assert response.status_code == 404


def test_get_nonexistent_trial_returns_404():
    response = client.get("/trials/NO_SUCH_TRIAL_XYZ")
    assert response.status_code == 404


def test_status_endpoint_returns_api_key():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert data["api"] == "running"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data


@pytest.mark.parametrize("path", ["/health", "/healthz", "/version", "/status", "/metrics"])
def test_meta_endpoints_return_200(path):
    response = client.get(path)
    assert response.status_code == 200


def test_root_version_is_correct():
    response = client.get("/")
    data = response.json()
    assert data["version"] == "1.2.0"


def test_patients_count_returns_integer():
    response = client.get("/patients/count")
    assert response.status_code == 200
    assert isinstance(response.json()["count"], int)


def test_trials_count_returns_integer():
    response = client.get("/trials/count")
    assert response.status_code == 200
    assert isinstance(response.json()["count"], int)


def test_list_patients_zero_limit_invalid():
    response = client.get("/patients?limit=0")
    assert response.status_code == 422


def test_list_trials_negative_limit_invalid():
    response = client.get("/trials?limit=-1")
    assert response.status_code == 422


def test_ping_returns_pong():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["ping"] == "pong"
