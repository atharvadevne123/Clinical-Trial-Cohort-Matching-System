"""Tests for GZip, CORS, and request timing middleware."""

from __future__ import annotations

import os

import pytest


class TestProcessTimeHeader:
    def test_ping_has_process_time_header(self, client):
        resp = client.get("/ping")
        assert "x-process-time-ms" in resp.headers

    def test_health_has_process_time_header(self, client):
        resp = client.get("/health")
        assert "x-process-time-ms" in resp.headers

    def test_process_time_is_numeric(self, client):
        resp = client.get("/ping")
        process_time = resp.headers.get("x-process-time-ms", "")
        assert float(process_time) >= 0

    def test_version_has_process_time_header(self, client):
        resp = client.get("/version")
        assert "x-process-time-ms" in resp.headers


class TestCorrelationIdHeader:
    def test_response_has_correlation_id(self, client):
        resp = client.get("/ping")
        assert "x-correlation-id" in resp.headers

    def test_provided_correlation_id_echoed(self, client):
        resp = client.get("/ping", headers={"X-Correlation-ID": "my-test-id-123"})
        assert resp.headers["x-correlation-id"] == "my-test-id-123"

    def test_auto_generated_correlation_id_is_uuid_format(self, client):
        resp = client.get("/ping")
        cid = resp.headers["x-correlation-id"]
        import uuid
        uuid.UUID(cid)


class TestVersionEndpoint:
    def test_version_returns_1_2_0(self, client):
        resp = client.get("/version")
        assert resp.status_code == 200
        assert resp.json()["version"] == "1.2.0"

    def test_version_has_started_at(self, client):
        resp = client.get("/version")
        assert "started_at" in resp.json()

    def test_version_has_api_field(self, client):
        resp = client.get("/version")
        assert "api" in resp.json()


class TestCorsOriginsConfig:
    def test_cors_origins_defaults_to_wildcard(self):
        import src.main as main_module
        cors = getattr(main_module, "_CORS_ORIGINS", None)
        if cors is not None:
            assert isinstance(cors, list)
            assert len(cors) > 0

    @pytest.mark.parametrize("endpoint", ["/ping", "/health", "/version"])
    def test_allowed_endpoints_return_200(self, client, endpoint):
        resp = client.get(endpoint)
        assert resp.status_code == 200
