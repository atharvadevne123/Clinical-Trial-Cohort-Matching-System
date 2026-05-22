"""Extended tests for the recruitment engine error handling."""

from __future__ import annotations

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from src.recruitment import RecruitmentEngine


class TestSendRecruitmentEmailErrors:
    CANDIDATE = {
        "patient_id": "P001",
        "patient_name": "Alice Smith",
        "email": "alice@example.com",
        "score": 0.85,
        "confidence": "HIGH",
        "recommendation": "Strong candidate",
        "trial_id": "T001",
        "trial_name": "Hypertension Study",
    }

    def test_smtp_exception_returns_false(self):
        engine = RecruitmentEngine(smtp_host="localhost", smtp_port=9999)
        with patch("src.recruitment.smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.__enter__.side_effect = smtplib.SMTPException("refused")
            result = engine.send_recruitment_email(self.CANDIDATE)
        assert result is False

    def test_os_error_returns_false(self):
        engine = RecruitmentEngine(smtp_host="bad-host", smtp_port=9999)
        with patch("src.recruitment.smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.__enter__.side_effect = OSError("no route to host")
            result = engine.send_recruitment_email(self.CANDIDATE)
        assert result is False

    def test_successful_send_returns_true(self):
        engine = RecruitmentEngine()
        with patch("src.recruitment.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = lambda s: mock_server
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            result = engine.send_recruitment_email(self.CANDIDATE)
        assert result is True

    def test_email_subject_contains_trial_name(self):
        engine = RecruitmentEngine()
        captured = {}
        with patch("src.recruitment.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_server.send_message.side_effect = lambda msg: captured.update({"subject": msg["Subject"]})
            mock_smtp.return_value.__enter__ = lambda s: mock_server
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            engine.send_recruitment_email(self.CANDIDATE)
        assert "Hypertension Study" in captured["subject"]

    def test_smtp_timeout_applied(self):
        engine = RecruitmentEngine()
        with patch("src.recruitment.smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.__enter__ = lambda s: MagicMock()
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            engine.send_recruitment_email(self.CANDIDATE)
        call_kwargs = mock_smtp.call_args
        assert call_kwargs is not None

    @pytest.mark.parametrize("score,expected_pct", [
        (1.0, "100%"),
        (0.5, "50%"),
        (0.0, "0%"),
    ])
    def test_email_body_contains_match_probability(self, score, expected_pct):
        engine = RecruitmentEngine()
        candidate = {**self.CANDIDATE, "score": score}
        captured = {}
        with patch("src.recruitment.smtplib.SMTP") as mock_smtp:
            def capture(msg):
                captured["body"] = msg.get_payload()
            mock_server = MagicMock()
            mock_server.send_message.side_effect = capture
            mock_smtp.return_value.__enter__ = lambda s: mock_server
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            engine.send_recruitment_email(candidate)
        assert expected_pct in captured.get("body", "")
