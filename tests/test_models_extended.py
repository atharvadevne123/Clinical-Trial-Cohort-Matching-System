"""Extended tests for ORM models — MatchStatus enum and model fields."""

from __future__ import annotations

import pytest

from src.models import MatchStatus


class TestMatchStatusEnum:
    def test_all_status_values_present(self):
        values = {s.value for s in MatchStatus}
        assert "PENDING" in values
        assert "ELIGIBLE" in values
        assert "INELIGIBLE" in values
        assert "ENROLLED" in values
        assert "WITHDRAWN" in values

    def test_match_status_is_str_subclass(self):
        assert isinstance(MatchStatus.ELIGIBLE, str)

    def test_match_status_value_equals_string(self):
        assert MatchStatus.ELIGIBLE == "ELIGIBLE"
        assert MatchStatus.INELIGIBLE == "INELIGIBLE"

    @pytest.mark.parametrize("status", ["PENDING", "ELIGIBLE", "INELIGIBLE", "ENROLLED", "WITHDRAWN"])
    def test_all_statuses_constructable_from_string(self, status):
        assert MatchStatus(status).value == status

    def test_invalid_status_raises_value_error(self):
        with pytest.raises(ValueError):
            MatchStatus("UNKNOWN_STATUS")

    def test_match_status_count(self):
        assert len(list(MatchStatus)) == 5


class TestPatientModelDefaults:
    def test_patient_created_at_is_set(self, db_session):
        from src.models import Patient
        p = Patient(id="MEXT_001", first_name="A", last_name="B", gender="male",
                    conditions=[], medications=[], allergies=[])
        db_session.add(p)
        db_session.commit()
        assert p.created_at is not None
        db_session.delete(p)
        db_session.commit()

    def test_trial_default_status_is_recruiting(self, db_session):
        from src.models import Trial
        t = Trial(id="MEXT_T001", name="Test", phase="Phase 1",
                  primary_condition="X", target_enrollment=10,
                  inclusion_criteria=[], exclusion_criteria=[])
        db_session.add(t)
        db_session.commit()
        assert t.status == "RECRUITING"
        db_session.delete(t)
        db_session.commit()
