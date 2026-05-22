"""Recruitment Engine — score candidates and dispatch outreach emails.

Provides async patient scoring against ML enrollment predictions and
SMTP-based outreach email dispatch for clinical trial recruitment.
"""

import asyncio
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, TypedDict

from src.ml_prediction import EnrollmentPredictor, predictor
from src.models import Patient, PatientTrialMatch, SessionLocal, Trial

logger = logging.getLogger(__name__)


class CandidateResult(TypedDict):
    """Typed dict for a scored candidate returned by score_eligible_patients."""

    patient_id: str
    patient_name: str
    email: str
    score: float
    confidence: str
    recommendation: str
    trial_id: str
    trial_name: str


_SMTP_HOST: str = os.environ.get("SMTP_HOST", "localhost")
_SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "1025"))
_SMTP_FROM: str = os.environ.get("SMTP_FROM", "noreply@trial.local")
_SMTP_TIMEOUT: int = int(os.environ.get("SMTP_TIMEOUT", "5"))


def _patient_to_dict(p: Patient) -> Dict[str, Any]:
    """Convert a Patient ORM object to the dict format expected by the predictor.

    Args:
        p: SQLAlchemy Patient model instance.

    Returns:
        Dict with id, date_of_birth, gender, conditions, and medications.
    """
    return {
        "id": str(p.id),
        "date_of_birth": p.date_of_birth,
        "gender": p.gender,
        "conditions": p.conditions or [],
        "medications": p.medications or [],
    }


class RecruitmentEngine:
    """Engine that scores eligible patients and sends recruitment outreach emails.

    Attributes:
        smtp_host: SMTP server hostname (default from SMTP_HOST env var).
        smtp_port: SMTP server port (default from SMTP_PORT env var).
    """

    def __init__(
        self,
        smtp_host: str = _SMTP_HOST,
        smtp_port: int = _SMTP_PORT,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    async def score_eligible_patients(
        self, trial_id: str, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Score all un-enrolled patients against a trial and return above-threshold candidates.

        Args:
            trial_id: ID of the trial to score against.
            threshold: Minimum enrollment_probability (0.0–1.0) to include a candidate.

        Returns:
            List of candidate dicts sorted by score descending.
        """
        with SessionLocal() as db:
            trial = db.query(Trial).filter(Trial.id == trial_id).first()
            if not trial:
                logger.warning("Trial %s not found.", trial_id)
                return []

            patients = db.query(Patient).all()
            scored: List[Dict[str, Any]] = []

            for p in patients:
                already_enrolled = db.query(PatientTrialMatch).filter(
                    PatientTrialMatch.patient_id == p.id,
                    PatientTrialMatch.trial_id == trial_id,
                    PatientTrialMatch.enrolled == True,  # noqa: E712
                ).first()
                if already_enrolled:
                    continue

                features = EnrollmentPredictor._dict_to_features(_patient_to_dict(p))
                result = predictor.predict(features, str(p.id), trial_id)

                if result.enrollment_probability >= threshold:
                    scored.append({
                        "patient_id": p.id,
                        "patient_name": f"{p.first_name} {p.last_name}",
                        "email": p.email or f"patient-{p.id}@trial.local",
                        "score": result.enrollment_probability,
                        "confidence": result.confidence,
                        "recommendation": result.recommendation,
                        "trial_id": trial_id,
                        "trial_name": trial.name,
                    })

        return sorted(scored, key=lambda x: (-x["score"], x["patient_id"]))

    def send_recruitment_email(self, candidate: Dict[str, Any]) -> bool:
        """Send a recruitment invitation email to a single candidate.

        Args:
            candidate: Dict with patient_id, patient_name, email, score,
                confidence, recommendation, trial_id, and trial_name.

        Returns:
            True if the email was sent successfully, False otherwise.
        """
        subject = f"Invitation: {candidate['trial_name']} Clinical Trial"
        body = (
            f"Dear {candidate['patient_name']},\n\n"
            f"Based on your health profile, you may be a strong match for our "
            f"{candidate['trial_name']} clinical trial.\n\n"
            f"Match probability: {candidate['score']:.0%}\n"
            f"Confidence: {candidate['confidence']}\n\n"
            f"To learn more or express interest, please contact your care team "
            f"and reference trial ID: {candidate['trial_id']}.\n\n"
            f"Best regards,\n"
            f"Clinical Trial Cohort Team\n"
        )
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = _SMTP_FROM
            msg["To"] = candidate["email"]
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=_SMTP_TIMEOUT) as server:
                server.send_message(msg)
            logger.info("Recruitment email sent to patient %s", candidate["patient_id"])
            return True
        except smtplib.SMTPException as exc:
            logger.warning("SMTP error for patient %s: %s", candidate["patient_id"], exc)
            return False
        except OSError as exc:
            logger.warning("Network error sending email to patient %s: %s", candidate["patient_id"], exc)
            return False

    async def run_recruitment_batch(
        self,
        trial_id: str,
        threshold: float = 0.6,
        max_recruits: int = 10,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Score patients, optionally send emails, and return a summary report.

        Args:
            trial_id: ID of the trial to recruit for.
            threshold: Minimum enrollment probability to include a candidate.
            max_recruits: Maximum number of candidates to process.
            dry_run: If True, score but do not send emails.

        Returns:
            Summary dict with trial_id, timestamp, counts, and per-candidate results.
        """
        candidates = await self.score_eligible_patients(trial_id, threshold)
        candidates = candidates[:max_recruits]

        results: Dict[str, Any] = {
            "trial_id": trial_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold,
            "candidates_scored": len(candidates),
            "emails_sent": 0,
            "dry_run": dry_run,
            "recruitment_results": [],
        }

        for candidate in candidates:
            sent = False
            if not dry_run:
                sent = self.send_recruitment_email(candidate)
                if sent:
                    results["emails_sent"] += 1

            results["recruitment_results"].append({
                "patient_id": candidate["patient_id"],
                "patient_name": candidate["patient_name"],
                "score": round(candidate["score"], 4),
                "confidence": candidate["confidence"],
                "email_sent": sent,
            })

        return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

async def _main() -> None:
    """CLI entry point: dry-run recruitment batch for a demo trial."""
    logging.basicConfig(level=logging.INFO)
    engine = RecruitmentEngine()
    results = await engine.run_recruitment_batch(
        trial_id="TRIAL_AFIB_001",
        threshold=0.5,
        max_recruits=10,
        dry_run=True,
    )
    import json
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(_main())
