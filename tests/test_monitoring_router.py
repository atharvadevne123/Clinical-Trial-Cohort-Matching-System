"""Tests for monitoring router endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import Base
from src.main import app, get_db

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

# Register monitoring router
from src.monitoring_router import router as monitoring_router
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
