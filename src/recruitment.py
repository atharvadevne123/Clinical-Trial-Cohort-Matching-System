"""Recruitment Engine — score candidates and dispatch outreach emails."""

import asyncio
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from src.models import Patient, Trial, PatientTrialMatch, SessionLocal
from src.ml_prediction import predictor, EnrollmentPredictor

logger = logging.getLogger(__name__)


def _patient_to_dict(p: Patient) -> Dict[str, Any]:
    """Convert a Patient ORM object to the dict format expected by the predictor."""
    return {
        "id": str(p.id),
        "date_of_birth": p.date_of_birth,
        "gender": p.gender,
        "conditions": p.conditions or [],
        "medications": p.medications or [],
    }


class RecruitmentEngine:
    def __init__(self, smtp_host: str = "localhost", smtp_port: int = 1025):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    async def score_eligible_patients(
        self, trial_id: str, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Return patients above the enrollment probability threshold, sorted by score."""
        with SessionLocal() as db:
            trial = db.query(Trial).filter(Trial.id == trial_id).first()
            if not trial:
                logger.warning(f"Trial {trial_id} not found.")
                return []

            patients = db.query(Patient).all()
            scored: List[Dict[str, Any]] = []

            for p in patients:
                # Skip patients already enrolled in this trial
                already_enrolled = db.query(PatientTrialMatch).filter(
                    PatientTrialMatch.patient_id == p.id,
                    PatientTrialMatch.trial_id == trial_id,
                    PatientTrialMatch.enrolled == True,
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

        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def send_recruitment_email(self, candidate: Dict[str, Any]) -> bool:
        """Send a recruitment email to a candidate. Returns True on success."""
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
            msg["From"] = "noreply@trial.local"
            msg["To"] = candidate["email"]
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=5) as server:
                server.send_message(msg)
            logger.info(f"Recruitment email sent to {candidate['patient_id']}")
            return True
        except Exception as e:
            logger.warning(f"Email failed for {candidate['patient_id']}: {e}")
            return False

    async def run_recruitment_batch(
        self,
        trial_id: str,
        threshold: float = 0.6,
        max_recruits: int = 10,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Score patients, optionally send emails, and return a summary."""
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

async def _main():
    logging.basicConfig(level=logging.INFO)
    engine = RecruitmentEngine()
    # Dry run to show candidates without sending emails
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
