"""Tests for new endpoints: DELETE, PATCH, bulk create, filters, and list matches."""

from __future__ import annotations

import pytest

PATIENT_BASE = {
    "id": "NE_P001",
    "first_name": "New",
    "last_name": "Endpoint",
    "date_of_birth": "1985-01-01T00:00:00Z",
    "gender": "male",
    "conditions": [{"code": "I10", "name": "Hypertension"}],
    "medications": [],
    "allergies": [],
}

TRIAL_BASE = {
    "id": "NE_T001",
    "name": "New Endpoint Trial",
    "phase": "Phase 2",
    "primary_condition": "Hypertension",
    "target_enrollment": 50,
    "inclusion_criteria": [],
    "exclusion_criteria": [],
}


@pytest.fixture(autouse=True)
def cleanup(client, db_session):
    from src.models import Patient, Trial

    yield
    db_session.query(Patient).filter(Patient.id.like("NE_%")).delete(synchronize_session=False)
    db_session.query(Trial).filter(Trial.id.like("NE_%")).delete(synchronize_session=False)
    db_session.commit()


class TestDeletePatient:
    def test_delete_existing_patient(self, client):
        client.post("/patients", json=PATIENT_BASE)
        resp = client.delete(f"/patients/{PATIENT_BASE['id']}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == PATIENT_BASE["id"]

    def test_delete_returns_404_when_not_found(self, client):
        resp = client.delete("/patients/NO_SUCH_NE_PATIENT")
        assert resp.status_code == 404

    def test_deleted_patient_no_longer_retrievable(self, client):
        client.post("/patients", json=PATIENT_BASE)
        client.delete(f"/patients/{PATIENT_BASE['id']}")
        resp = client.get(f"/patients/{PATIENT_BASE['id']}")
        assert resp.status_code == 404


class TestDeleteTrial:
    def test_delete_existing_trial(self, client):
        client.post("/trials", json=TRIAL_BASE)
        resp = client.delete(f"/trials/{TRIAL_BASE['id']}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == TRIAL_BASE["id"]

    def test_delete_returns_404_when_not_found(self, client):
        resp = client.delete("/trials/NO_SUCH_NE_TRIAL")
        assert resp.status_code == 404


class TestPatchPatient:
    def test_patch_first_name(self, client):
        client.post("/patients", json=PATIENT_BASE)
        resp = client.patch(f"/patients/{PATIENT_BASE['id']}", json={"first_name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["first_name"] == "Updated"

    def test_patch_gender(self, client):
        client.post("/patients", json=PATIENT_BASE)
        resp = client.patch(f"/patients/{PATIENT_BASE['id']}", json={"gender": "female"})
        assert resp.status_code == 200
        assert resp.json()["gender"] == "female"

    def test_patch_returns_404_when_not_found(self, client):
        resp = client.patch("/patients/NO_SUCH_NE_PATIENT", json={"first_name": "X"})
        assert resp.status_code == 404

    def test_patch_preserves_unspecified_fields(self, client):
        client.post("/patients", json=PATIENT_BASE)
        client.patch(f"/patients/{PATIENT_BASE['id']}", json={"first_name": "Changed"})
        resp = client.get(f"/patients/{PATIENT_BASE['id']}")
        assert resp.json()["last_name"] == PATIENT_BASE["last_name"]


class TestPatchTrial:
    def test_patch_trial_name(self, client):
        client.post("/trials", json=TRIAL_BASE)
        resp = client.patch(f"/trials/{TRIAL_BASE['id']}", json={"name": "Renamed Trial"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed Trial"

    def test_patch_trial_returns_404_when_not_found(self, client):
        resp = client.patch("/trials/NO_SUCH_NE_TRIAL", json={"name": "X"})
        assert resp.status_code == 404


class TestPatientFilters:
    def test_filter_by_gender_returns_only_matching(self, client):
        client.post("/patients", json=PATIENT_BASE)
        resp = client.get("/patients?gender=male")
        assert resp.status_code == 200
        for p in resp.json():
            assert p["gender"] == "male"

    def test_filter_by_unknown_gender_returns_empty_or_subset(self, client):
        resp = client.get("/patients?gender=nonbinary_ne")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_condition_filter_query_param_accepted(self, client):
        resp = client.get("/patients?condition=I10")
        assert resp.status_code == 200


class TestTrialFilters:
    def test_filter_by_phase(self, client):
        client.post("/trials", json=TRIAL_BASE)
        resp = client.get("/trials?phase=Phase+2")
        assert resp.status_code == 200
        for t in resp.json():
            assert t["phase"] == "Phase 2"

    def test_filter_by_status_accepted(self, client):
        resp = client.get("/trials?status=RECRUITING")
        assert resp.status_code == 200


class TestBulkPatients:
    def test_bulk_create_two_patients(self, client):
        p2 = {**PATIENT_BASE, "id": "NE_P002", "first_name": "Bulk2"}
        resp = client.post("/patients/bulk", json=[PATIENT_BASE, p2])
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] == 2
        assert data["skipped"] == 0

    def test_bulk_create_skips_duplicates(self, client):
        client.post("/patients", json=PATIENT_BASE)
        p2 = {**PATIENT_BASE, "id": "NE_P002", "first_name": "Bulk2"}
        resp = client.post("/patients/bulk", json=[PATIENT_BASE, p2])
        data = resp.json()
        assert data["skipped"] == 1
        assert PATIENT_BASE["id"] in data["skipped_ids"]

    def test_bulk_create_exceeding_limit_returns_400(self, client):
        patients = [{**PATIENT_BASE, "id": f"NE_BULK_{i}"} for i in range(101)]
        resp = client.post("/patients/bulk", json=patients)
        assert resp.status_code == 400


class TestBulkTrials:
    def test_bulk_create_trials(self, client):
        t2 = {**TRIAL_BASE, "id": "NE_T002", "name": "Bulk Trial 2"}
        resp = client.post("/trials/bulk", json=[TRIAL_BASE, t2])
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] == 2

    def test_bulk_trials_exceeding_limit_returns_400(self, client):
        trials = [{**TRIAL_BASE, "id": f"NE_BTRIAL_{i}", "name": f"Trial {i}"} for i in range(51)]
        resp = client.post("/trials/bulk", json=trials)
        assert resp.status_code == 400


class TestListAllMatches:
    def test_list_matches_returns_dict_with_total(self, client):
        resp = client.get("/matches")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "matches" in data

    def test_list_matches_status_filter_accepted(self, client):
        resp = client.get("/matches?status=ELIGIBLE")
        assert resp.status_code == 200

    @pytest.mark.parametrize("limit", [1, 10, 50])
    def test_list_matches_limit_param(self, client, limit):
        resp = client.get(f"/matches?limit={limit}")
        assert resp.status_code == 200
        assert len(resp.json()["matches"]) <= limit


class TestPingEndpoint:
    def test_ping_returns_pong(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json() == {"ping": "pong"}

    def test_ping_has_process_time_header(self, client):
        resp = client.get("/ping")
        assert "x-process-time-ms" in resp.headers
