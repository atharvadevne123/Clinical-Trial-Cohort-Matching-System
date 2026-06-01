"""Tests for monitoring router endpoints."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app, get_db
from src.models import Base
from src.monitoring_router import router as monitoring_router

TEST_DB = "sqlite:///./test_monitoring_router.db"
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
app.include_router(monitoring_router)

client = TestClient(app)


def test_monitoring_drift_endpoint():
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data


def test_monitoring_summary_endpoint():
    response = client.get("/monitoring/summary")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data


def test_monitoring_reset_endpoint():
    response = client.post("/monitoring/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "reset"


def test_drift_after_reset_has_zero_samples():
    client.post("/monitoring/reset")
    response = client.get("/monitoring/summary")
    data = response.json()
    assert data["count"] == 0


def test_drift_endpoint_with_api_key_when_not_required(monkeypatch):
    """When API_KEY is not set, requests without a key should still succeed."""
    monkeypatch.delenv("API_KEY", raising=False)
    response = client.get("/monitoring/drift")
    assert response.status_code == 200


def test_monitoring_drift_returns_drift_detected_key():
    response = client.get("/monitoring/drift")
    data = response.json()
    assert isinstance(data.get("drift_detected"), bool)


def test_monitoring_summary_min_max_none_when_empty():
    """When no predictions recorded, min/max should be None."""
    client.post("/monitoring/reset")
    response = client.get("/monitoring/summary")
    data = response.json()
    if data["count"] == 0:
        assert data["mean"] is None
        assert data["min"] is None
        assert data["max"] is None


def test_set_reference_valid_payload():
    payload = {"probabilities": [0.3, 0.5, 0.7, 0.4, 0.6]}
    response = client.post("/monitoring/set-reference", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["reference_samples"] == 5


def test_set_reference_invalid_payload():
    payload = {"probabilities": "not-a-list"}
    response = client.post("/monitoring/set-reference", json=payload)
    assert response.status_code == 400


def test_set_reference_empty_list():
    payload = {"probabilities": []}
    response = client.post("/monitoring/set-reference", json=payload)
    assert response.status_code == 200
    assert response.json()["reference_samples"] == 0


def test_monitoring_reset_message():
    response = client.post("/monitoring/reset")
    data = response.json()
    assert "message" in data
