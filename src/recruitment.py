import asyncio
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from sqlalchemy import select
from src.models import Patient, Trial, Enrollment
from src.database import SessionLocal
from src.ml_prediction import EnrollmentPredictor

class RecruitmentEngine:
    def __init__(self, smtp_host="localhost", smtp_port=1025):
        self.predictor = EnrollmentPredictor()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.db = SessionLocal()

    async def score_eligible_patients(self, trial_id: int, threshold: float = 0.5):
        trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
        if not trial:
            return []

        patients = self.db.query(Patient).all()
        scored = []
        
        for p in patients:
            enrolled = self.db.query(Enrollment).filter(
                Enrollment.patient_id == p.id,
                Enrollment.trial_id == trial_id
            ).first()
            if enrolled:
                continue

            features = {
                'age': p.age,
                'gender_male': 1 if p.gender == 'M' else 0,
                'num_conditions': p.num_conditions,
                'num_medications': p.num_medications,
                'has_diabetes': p.has_diabetes,
                'has_hypertension': p.has_hypertension,
                'has_heart_disease': p.has_heart_disease,
                'has_cancer': p.has_cancer,
                'has_afib': p.has_afib,
                'smoker': p.smoker,
                'bmi': p.bmi,
                'prior_trial_participation': p.prior_trial_participation,
                'distance_to_site_km': p.distance_to_site_km,
                'num_exclusion_flags': 0
            }
            
            prob = self.predictor.predict([features])[0]
            if prob >= threshold:
                scored.append({
                    'patient_id': p.id,
                    'patient_name': f"Patient_{p.id}",
                    'email': f"patient{p.id}@trial.local",
                    'score': prob,
                    'trial_id': trial_id,
                    'trial_name': trial.name
                })
        
        return sorted(scored, key=lambda x: x['score'], reverse=True)

    def send_recruitment_email(self, patient, trial_name):
        subject = f"You May Be Eligible for {trial_name} Clinical Trial"
        body = f"""
Dear {patient['patient_name']},

Based on our analysis, you may be a good match for our {trial_name} clinical trial.

Match Score: {patient['score']:.2%}

We believe your health profile aligns well with this study. If interested, 
please visit: http://localhost:8000/enrollment/{patient['patient_id']}/{patient['trial_id']}

Best regards,
Clinical Trial Cohort Team
        """
        
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = 'noreply@trial.local'
            msg['To'] = patient['email']
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Email failed: {e}")
            return False

    async def run_recruitment_batch(self, trial_id: int, threshold: float = 0.6, max_recruits: int = 10):
        candidates = await self.score_eligible_patients(trial_id, threshold)
        candidates = candidates[:max_recruits]
        
        results = {
            'trial_id': trial_id,
            'timestamp': datetime.now().isoformat(),
            'candidates_scored': len(candidates),
            'emails_sent': 0,
            'recruitment_results': []
        }
        
        for candidate in candidates:
            sent = self.send_recruitment_email(candidate, candidate['trial_name'])
            results['recruitment_results'].append({
                'patient_id': candidate['patient_id'],
                'score': candidate['score'],
                'email_sent': sent
            })
            if sent:
                results['emails_sent'] += 1
        
        return results

async def main():
    engine = RecruitmentEngine()
    results = await engine.run_recruitment_batch(trial_id=1, threshold=0.5, max_recruits=5)
    print(results)

if __name__ == '__main__':
    asyncio.run(main())
