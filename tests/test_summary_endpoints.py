"""Tests for /summary, /readyz, and /patients/{id}/eligible-trials endpoints."""

from __future__ import annotations

import pytest


class TestSummaryEndpoint:
    def test_summary_returns_expected_keys(self, client):
        resp = client.get("/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "patients_by_gender" in data
        assert "trials_by_phase" in data
        assert "matches_by_status" in data

    def test_summary_patients_by_gender_is_dict(self, client):
        resp = client.get("/summary")
        assert isinstance(resp.json()["patients_by_gender"], dict)

    def test_summary_trials_by_phase_is_dict(self, client):
        resp = client.get("/summary")
        assert isinstance(resp.json()["trials_by_phase"], dict)

    def test_summary_counts_created_patient(self, client, sample_patient):
        resp = client.get("/summary")
        data = resp.json()
        assert data["patients_by_gender"].get("female", 0) >= 1


class TestReadyzEndpoint:
    def test_readyz_returns_200_when_db_ok(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200

    def test_readyz_has_status_key(self, client):
        resp = client.get("/readyz")
        assert "status" in resp.json()

    def test_readyz_has_database_key(self, client):
        resp = client.get("/readyz")
        assert "database" in resp.json()

    def test_readyz_database_is_ok_when_reachable(self, client):
        resp = client.get("/readyz")
        assert resp.json().get("database") == "ok"


class TestEligibleTrialsEndpoint:
    def test_eligible_trials_returns_404_for_unknown_patient(self, client):
        resp = client.get("/patients/NO_SUCH_NE_ET_PATIENT/eligible-trials")
        assert resp.status_code == 404

    def test_eligible_trials_returns_dict_with_required_keys(self, client, sample_patient):
        resp = client.get(f"/patients/{sample_patient.id}/eligible-trials")
        assert resp.status_code == 200
        data = resp.json()
        assert "patient_id" in data
        assert "eligible_trial_count" in data
        assert "trial_ids" in data

    def test_eligible_trials_empty_when_no_matches(self, client, sample_patient):
        resp = client.get(f"/patients/{sample_patient.id}/eligible-trials")
        assert resp.status_code == 200
        assert resp.json()["eligible_trial_count"] == 0

    def test_eligible_trials_patient_id_matches_request(self, client, sample_patient):
        resp = client.get(f"/patients/{sample_patient.id}/eligible-trials")
        assert resp.json()["patient_id"] == sample_patient.id


class TestPingEndpointExtended:
    def test_ping_is_fast(self, client):
        resp = client.get("/ping")
        process_time = float(resp.headers.get("x-process-time-ms", 0))
        assert process_time < 500

    def test_ping_no_db_dependency(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200
