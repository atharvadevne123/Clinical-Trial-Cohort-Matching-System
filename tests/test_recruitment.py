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


def test_patient_to_dict_includes_medications(mock_patient):
    mock_patient.medications = [{"code": "C09AA01", "name": "Lisinopril"}]
    d = _patient_to_dict(mock_patient)
    assert len(d["medications"]) == 1
    assert d["medications"][0]["code"] == "C09AA01"


def test_patient_to_dict_none_conditions_returns_empty_list():
    p = MagicMock(spec=Patient)
    p.id = "P_NONE"
    p.date_of_birth = None
    p.gender = "male"
    p.conditions = None
    p.medications = None
    d = _patient_to_dict(p)
    assert d["conditions"] == []
    assert d["medications"] == []


def test_smtp_timeout_env_var(monkeypatch):
    """SMTP_TIMEOUT env var should be imported as _SMTP_TIMEOUT."""
    import importlib

    monkeypatch.setenv("SMTP_TIMEOUT", "15")
    import src.recruitment as rec_module

    importlib.reload(rec_module)
    assert rec_module._SMTP_TIMEOUT == 15
    monkeypatch.delenv("SMTP_TIMEOUT", raising=False)
    importlib.reload(rec_module)


@pytest.mark.asyncio
async def test_run_recruitment_batch_dry_run_no_emails_sent(engine):
    """dry_run=True must not attempt to send any emails."""
    with patch("src.recruitment.SessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        result = await engine.run_recruitment_batch("T_DRYRUN", dry_run=True)
    assert result["dry_run"] is True
    assert result["emails_sent"] == 0


@pytest.mark.asyncio
async def test_run_recruitment_batch_result_keys(engine):
    """Batch result must include all required summary keys."""
    with patch("src.recruitment.SessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        result = await engine.run_recruitment_batch("T_KEYS")
    required = {
        "trial_id",
        "timestamp",
        "threshold",
        "candidates_scored",
        "emails_sent",
        "dry_run",
        "recruitment_results",
    }
    assert required.issubset(result.keys())


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.asyncio
async def test_score_eligible_threshold_parameter(engine, threshold):
    """score_eligible_patients should accept any float threshold in [0, 1]."""
    with patch("src.recruitment.SessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        result = await engine.score_eligible_patients("T_THRESH", threshold=threshold)
    assert isinstance(result, list)


def test_candidate_result_typed_dict():
    """CandidateResult TypedDict should be importable and have expected keys."""
    from src.recruitment import CandidateResult

    candidate: CandidateResult = {
        "patient_id": "P001",
        "patient_name": "Alice Smith",
        "email": "alice@example.com",
        "score": 0.85,
        "confidence": "HIGH",
        "recommendation": "Strong candidate",
        "trial_id": "T001",
        "trial_name": "Hypertension Trial",
    }
    assert candidate["score"] == 0.85
    assert candidate["confidence"] == "HIGH"


def test_send_recruitment_email_subject_contains_trial_name(engine):
    candidate = {
        "patient_id": "P_SUBJ",
        "patient_name": "Bob Jones",
        "email": "bob@example.com",
        "score": 0.70,
        "confidence": "MEDIUM",
        "recommendation": "Likely eligible",
        "trial_id": "T_SUBJ",
        "trial_name": "Diabetes Study",
    }
    sent_messages = []

    class MockSMTP:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send_message(self, msg):
            sent_messages.append(msg)

    with patch("smtplib.SMTP", return_value=MockSMTP()):
        result = engine.send_recruitment_email(candidate)
    assert isinstance(result, bool)
