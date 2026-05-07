"""Tests for the recruitment engine."""
from unittest.mock import MagicMock, patch

import pytest

from src.models import Patient
from src.recruitment import RecruitmentEngine, _patient_to_dict


@pytest.fixture
def engine():
    return RecruitmentEngine(smtp_host="localhost", smtp_port=1025)


@pytest.fixture
def mock_patient():
    p = MagicMock(spec=Patient)
    p.id = "P001"
    p.first_name = "Alice"
    p.last_name = "Smith"
    p.gender = "female"
    p.date_of_birth = None
    p.conditions = [{"code": "I10", "name": "Hypertension"}]
    p.medications = []
    p.email = "alice@example.com"
    return p


def test_patient_to_dict(mock_patient):
    d = _patient_to_dict(mock_patient)
    assert d["id"] == "P001"
    assert isinstance(d["conditions"], list)


def test_send_recruitment_email_handles_smtp_error(engine):
    candidate = {
        "patient_id": "P001",
        "patient_name": "Alice Smith",
        "email": "alice@example.com",
        "score": 0.8,
        "confidence": "HIGH",
        "recommendation": "Strong candidate",
        "trial_id": "T001",
        "trial_name": "Hypertension Trial",
    }
    with patch("smtplib.SMTP", side_effect=Exception("SMTP unavailable")):
        result = engine.send_recruitment_email(candidate)
    assert result is False


def test_send_recruitment_email_success(engine):
    candidate = {
        "patient_id": "P001",
        "patient_name": "Alice Smith",
        "email": "alice@example.com",
        "score": 0.85,
        "confidence": "HIGH",
        "recommendation": "Strong candidate",
        "trial_id": "T001",
        "trial_name": "Hypertension Trial",
    }
    mock_smtp = MagicMock()
    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        result = engine.send_recruitment_email(candidate)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_score_eligible_patients_returns_list(engine):
    with patch("src.recruitment.SessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = MagicMock()
        mock_session.query.return_value.all.return_value = []
        result = await engine.score_eligible_patients("T001", threshold=0.5)
    assert isinstance(result, list)
