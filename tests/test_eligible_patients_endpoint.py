"""Tests for GET /trials/{id}/eligible-patients endpoint."""

from __future__ import annotations

import pytest


class TestEligiblePatientsEndpoint:
    def test_nonexistent_trial_returns_404(self, client):
        resp = client.get("/trials/NO_SUCH_TRIAL_XYZ/eligible-patients")
        assert resp.status_code == 404

    def test_404_has_detail_key(self, client):
        resp = client.get("/trials/NO_SUCH_TRIAL_XYZ/eligible-patients")
        data = resp.json()
        assert "detail" in data

    def test_valid_trial_returns_200(self, client, db_session):
        from src.models import Trial

        trial = Trial(
            id="T_EP_001",
            name="EP Trial",
            phase="2",
            primary_condition="Hypertension",
            status="recruiting",
            inclusion_criteria=[],
            exclusion_criteria=[],
        )
        db_session.add(trial)
        db_session.commit()

        resp = client.get("/trials/T_EP_001/eligible-patients")
        assert resp.status_code == 200

    def test_response_has_trial_id(self, client, db_session):
        from src.models import Trial

        trial = Trial(
            id="T_EP_002",
            name="EP Trial 2",
            phase="1",
            primary_condition="Diabetes",
            status="recruiting",
            inclusion_criteria=[],
            exclusion_criteria=[],
        )
        db_session.add(trial)
        db_session.commit()

        resp = client.get("/trials/T_EP_002/eligible-patients")
        data = resp.json()
        assert data["trial_id"] == "T_EP_002"

    def test_response_has_eligible_patient_count(self, client, db_session):
        from src.models import Trial

        trial = Trial(
            id="T_EP_003",
            name="EP Trial 3",
            phase="3",
            primary_condition="Cancer",
            status="recruiting",
            inclusion_criteria=[],
            exclusion_criteria=[],
        )
        db_session.add(trial)
        db_session.commit()

        resp = client.get("/trials/T_EP_003/eligible-patients")
        data = resp.json()
        assert "eligible_patient_count" in data
        assert isinstance(data["eligible_patient_count"], int)

    def test_response_has_patient_ids_list(self, client, db_session):
        from src.models import Trial

        trial = Trial(
            id="T_EP_004",
            name="EP Trial 4",
            phase="2",
            primary_condition="Asthma",
            status="recruiting",
            inclusion_criteria=[],
            exclusion_criteria=[],
        )
        db_session.add(trial)
        db_session.commit()

        resp = client.get("/trials/T_EP_004/eligible-patients")
        data = resp.json()
        assert "patient_ids" in data
        assert isinstance(data["patient_ids"], list)

    def test_empty_trial_has_zero_eligible(self, client, db_session):
        from src.models import Trial

        trial = Trial(
            id="T_EP_005",
            name="EP Trial 5",
            phase="1",
            primary_condition="Rare Disease",
            status="recruiting",
            inclusion_criteria=[],
            exclusion_criteria=[],
        )
        db_session.add(trial)
        db_session.commit()

        resp = client.get("/trials/T_EP_005/eligible-patients")
        data = resp.json()
        assert data["eligible_patient_count"] == 0
        assert data["patient_ids"] == []

    @pytest.mark.parametrize("trial_id", ["", "NONE", "null", "undefined"])
    def test_various_invalid_ids_return_404(self, client, trial_id):
        resp = client.get(f"/trials/{trial_id}/eligible-patients")
        assert resp.status_code in (404, 405)
