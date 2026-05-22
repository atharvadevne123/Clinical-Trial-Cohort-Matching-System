"""Tests for custom 404 and 422 error handlers."""

from __future__ import annotations

import pytest


class TestCustomNotFoundHandler:
    def test_unknown_path_returns_404(self, client):
        resp = client.get("/this-path-does-not-exist-xyz")
        assert resp.status_code == 404

    def test_404_response_has_error_key(self, client):
        resp = client.get("/patients/ABSOLUTELY_NONEXISTENT_PATIENT_XYZ123")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data or "error" in data

    def test_nonexistent_trial_returns_404(self, client):
        resp = client.get("/trials/NO_SUCH_TRIAL_AT_ALL_XYZ")
        assert resp.status_code == 404


class TestCustomValidationErrorHandler:
    def test_invalid_limit_param_returns_422(self, client):
        resp = client.get("/patients?limit=0")
        assert resp.status_code == 422

    def test_invalid_skip_param_returns_422(self, client):
        resp = client.get("/patients?skip=-1")
        assert resp.status_code == 422

    def test_422_response_has_detail_or_error(self, client):
        resp = client.get("/patients?limit=0")
        data = resp.json()
        assert "detail" in data or "error" in data

    def test_missing_required_field_returns_422(self, client):
        resp = client.post("/patients", json={"first_name": "Only"})
        assert resp.status_code == 422

    @pytest.mark.parametrize("endpoint,params", [
        ("/patients", {"limit": 501}),
        ("/trials", {"limit": 501}),
    ])
    def test_limit_over_max_returns_422(self, client, endpoint, params):
        resp = client.get(endpoint, params=params)
        assert resp.status_code == 422


class TestAdminSeedEndpoint:
    def test_seed_endpoint_exists(self, client, db_session):
        from fastapi.testclient import TestClient
        from src.main import app
        safe_client = TestClient(app, raise_server_exceptions=False)
        resp = safe_client.post("/admin/seed")
        assert resp.status_code in (200, 403, 422, 500)

    def test_seed_route_is_registered(self, client):
        from src.main import app
        paths = [r.path for r in app.routes]
        assert "/admin/seed" in paths
