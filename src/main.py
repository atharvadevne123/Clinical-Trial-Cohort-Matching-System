"""Clinical Trial Cohort Matching API"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Security, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from src.models import init_db, Patient, Trial, PatientTrialMatch, SessionLocal
from src.schemas import PatientCreate, PatientResponse, TrialCreate, TrialResponse
from src.eligibility import matcher
from src.nlp import nlp_processor
from src.fhir import fhir_client
from src.ml_prediction import predictor, EnrollmentPredictor, create_ml_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

API_KEY = os.environ.get("API_KEY", "")  # empty = auth disabled (dev mode)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(_api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")
    return key


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database…")
    try:
        init_db()
        logger.info("Database ready.")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        raise
    yield


app = FastAPI(
    title="Clinical Trial Cohort Matching",
    description="Match patients to clinical trials using rule-based eligibility and ML enrollment prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(create_ml_router())

# ------------------------------------------------------------------
# DB dependency
# ------------------------------------------------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------
# Health / meta
# ------------------------------------------------------------------


@app.get("/", tags=["Meta"])
def root():
    return {"message": "Clinical Trial Cohort Matching API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "healthy"}


@app.get("/status", tags=["Meta"])
def status(db: Session = Depends(get_db)):
    patient_count = db.query(Patient).count()
    trial_count = db.query(Trial).count()
    match_count = db.query(PatientTrialMatch).count()
    return {
        "api": "running",
        "patients": patient_count,
        "trials": trial_count,
        "matches": match_count,
    }


# ------------------------------------------------------------------
# Patients
# ------------------------------------------------------------------


@app.post("/patients", response_model=PatientResponse, tags=["Patients"], dependencies=[Depends(require_api_key)])
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    if db.query(Patient).filter(Patient.id == patient.id).first():
        raise HTTPException(status_code=409, detail="Patient already exists")

    db_patient = Patient(
        id=patient.id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        date_of_birth=patient.date_of_birth,
        gender=patient.gender,
        email=patient.email,
        phone=patient.phone,
        postal_code=patient.postal_code,
        conditions=patient.conditions or [],
        medications=patient.medications or [],
        allergies=patient.allergies or [],
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@app.get("/patients/{patient_id}", response_model=PatientResponse, tags=["Patients"])
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/patients", response_model=list[PatientResponse], tags=["Patients"])
def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return db.query(Patient).offset(skip).limit(limit).all()


# ------------------------------------------------------------------
# Trials
# ------------------------------------------------------------------


@app.post("/trials", response_model=TrialResponse, tags=["Trials"], dependencies=[Depends(require_api_key)])
def create_trial(trial: TrialCreate, db: Session = Depends(get_db)):
    if db.query(Trial).filter(Trial.id == trial.id).first():
        raise HTTPException(status_code=409, detail="Trial already exists")

    db_trial = Trial(
        id=trial.id,
        name=trial.name,
        description=trial.description,
        sponsor=trial.sponsor,
        phase=trial.phase,
        primary_condition=trial.primary_condition,
        target_enrollment=trial.target_enrollment,
        inclusion_criteria=trial.inclusion_criteria or [],
        exclusion_criteria=trial.exclusion_criteria or [],
        start_date=trial.start_date,
        completion_date=trial.completion_date,
    )
    db.add(db_trial)
    db.commit()
    db.refresh(db_trial)
    return db_trial


@app.get("/trials/{trial_id}", response_model=TrialResponse, tags=["Trials"])
def get_trial(trial_id: str, db: Session = Depends(get_db)):
    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    return trial


@app.get("/trials", response_model=list[TrialResponse], tags=["Trials"])
def list_trials(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return db.query(Trial).offset(skip).limit(limit).all()


# ------------------------------------------------------------------
# Eligibility matching (rule-based + ML)
# ------------------------------------------------------------------


@app.post("/match/{patient_id}/{trial_id}", tags=["Matching"], dependencies=[Depends(require_api_key)])
def check_match(patient_id: str, trial_id: str, db: Session = Depends(get_db)):
    # Prevent duplicate match records
    existing = db.query(PatientTrialMatch).filter(
        PatientTrialMatch.patient_id == patient_id,
        PatientTrialMatch.trial_id == trial_id,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Match record already exists for this patient-trial pair")

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    patient_dict = {
        "id": patient.id,
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "date_of_birth": patient.date_of_birth,
        "gender": patient.gender,
        "conditions": patient.conditions or [],
        "medications": patient.medications or [],
    }
    trial_dict = {
        "id": trial.id,
        "inclusion_criteria": trial.inclusion_criteria or [],
        "exclusion_criteria": trial.exclusion_criteria or [],
    }

    # Rule-based eligibility
    rule_result = matcher.check_match(patient_dict, trial_dict)

    # ML enrollment prediction
    features = EnrollmentPredictor._dict_to_features(patient_dict)
    ml_result = predictor.predict(features, patient_id, trial_id)

    rule_score = float(rule_result["match_score"])          # 0–100
    ml_score = ml_result.enrollment_probability * 100       # 0–100
    combined = round(0.5 * rule_score + 0.5 * ml_score, 2)

    match = PatientTrialMatch(
        patient_id=patient_id,
        trial_id=trial_id,
        rule_match_score=rule_score,
        ml_match_score=ml_score,
        enrollment_probability=ml_result.enrollment_probability,
        combined_score=combined,
        match_status="ELIGIBLE" if rule_result["eligible"] else "INELIGIBLE",
        matched_criteria=rule_result["matched_inclusion"],
        violated_criteria=rule_result["violated_exclusion"],
        reasons=rule_result["reasons"],
    )
    db.add(match)
    db.commit()
    db.refresh(match)

    return {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "eligible": rule_result["eligible"],
        "rule_match_score": rule_score,
        "ml_match_score": round(ml_score, 2),
        "enrollment_probability": ml_result.enrollment_probability,
        "combined_score": combined,
        "confidence": ml_result.confidence,
        "recommendation": ml_result.recommendation,
        "matched_criteria": rule_result["matched_inclusion"],
        "violated_criteria": rule_result["violated_exclusion"],
        "reasons": rule_result["reasons"],
        "key_factors": ml_result.key_factors,
    }


@app.get("/matches/{trial_id}", tags=["Matching"])
def get_trial_matches(
    trial_id: str,
    status: Optional[str] = Query(None, description="Filter by match_status (ELIGIBLE/INELIGIBLE)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    if not db.query(Trial).filter(Trial.id == trial_id).first():
        raise HTTPException(status_code=404, detail="Trial not found")

    q = db.query(PatientTrialMatch).filter(PatientTrialMatch.trial_id == trial_id)
    if status:
        q = q.filter(PatientTrialMatch.match_status == status.upper())
    matches = q.offset(skip).limit(limit).all()

    total = db.query(PatientTrialMatch).filter(PatientTrialMatch.trial_id == trial_id).count()
    eligible = db.query(PatientTrialMatch).filter(
        PatientTrialMatch.trial_id == trial_id,
        PatientTrialMatch.match_status == "ELIGIBLE",
    ).count()

    return {
        "trial_id": trial_id,
        "total_matches": total,
        "eligible_count": eligible,
        "ineligible_count": total - eligible,
        "matches": matches,
    }


@app.get("/patients/{patient_id}/matches", tags=["Matching"])
def get_patient_matches(patient_id: str, db: Session = Depends(get_db)):
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=404, detail="Patient not found")
    matches = (
        db.query(PatientTrialMatch)
        .filter(PatientTrialMatch.patient_id == patient_id)
        .order_by(PatientTrialMatch.combined_score.desc())
        .all()
    )
    return {"patient_id": patient_id, "total_matches": len(matches), "matches": matches}


# ------------------------------------------------------------------
# NLP
# ------------------------------------------------------------------


from pydantic import BaseModel


class ClinicalNoteRequest(BaseModel):
    text: str


@app.post("/nlp/extract-entities", tags=["NLP"])
def extract_clinical_entities(request: ClinicalNoteRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    entities = nlp_processor.extract_entities(request.text)
    return {
        "text_length": len(request.text),
        "entities_found": (
            len(entities["conditions"]) + len(entities["medications"]) + len(entities["symptoms"])
        ),
        "extraction_result": entities,
    }


@app.post("/nlp/clinical-profile", tags=["NLP"])
def generate_clinical_profile(request: ClinicalNoteRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return {
        "clinical_profile": nlp_processor.summarize_clinical_profile(request.text),
        "text_length": len(request.text),
    }


@app.post("/patients/{patient_id}/analyze-notes", tags=["NLP"], dependencies=[Depends(require_api_key)])
def analyze_patient_notes(patient_id: str, request: ClinicalNoteRequest, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    entities = nlp_processor.extract_entities(request.text)

    if entities["conditions"]:
        patient.conditions = entities["conditions"]
    if entities["medications"]:
        patient.medications = entities["medications"]

    db.commit()
    db.refresh(patient)

    return {
        "patient_id": patient_id,
        "extracted_conditions": entities["conditions"],
        "extracted_medications": entities["medications"],
        "extracted_symptoms": entities["symptoms"],
        "patient_updated": True,
    }


# ------------------------------------------------------------------
# FHIR import
# ------------------------------------------------------------------


@app.post("/fhir/import/{fhir_patient_id}", tags=["FHIR"], dependencies=[Depends(require_api_key)])
def import_fhir_patient(fhir_patient_id: str, db: Session = Depends(get_db)):
    """Fetch a patient from the FHIR server and upsert into the local DB."""
    profile = fhir_client.fetch_complete_patient_profile(fhir_patient_id)

    patient = db.query(Patient).filter(Patient.id == fhir_patient_id).first()
    if patient:
        patient.conditions = profile.get("conditions", [])
        patient.medications = profile.get("medications", [])
    else:
        from datetime import datetime
        dob_str = profile.get("date_of_birth")
        dob = datetime.fromisoformat(dob_str) if dob_str else None
        patient = Patient(
            id=fhir_patient_id,
            first_name=profile.get("first_name", ""),
            last_name=profile.get("last_name", ""),
            date_of_birth=dob,
            gender=profile.get("gender", "unknown"),
            email=profile.get("email"),
            phone=profile.get("phone"),
            postal_code=profile.get("postal_code"),
            conditions=profile.get("conditions", []),
            medications=profile.get("medications", []),
        )
        db.add(patient)

    db.commit()
    db.refresh(patient)
    return {"imported": True, "patient_id": patient.id, "profile": profile}


# ------------------------------------------------------------------
# Recruitment
# ------------------------------------------------------------------


@app.get("/recruitment/candidates/{trial_id}", tags=["Recruitment"])
async def get_recruitment_candidates(
    trial_id: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum enrollment probability"),
    db: Session = Depends(get_db),
):
    """Score all un-enrolled patients against a trial and return ranked candidates."""
    if not db.query(Trial).filter(Trial.id == trial_id).first():
        raise HTTPException(status_code=404, detail="Trial not found")
    from src.recruitment import RecruitmentEngine
    engine = RecruitmentEngine()
    candidates = await engine.score_eligible_patients(trial_id, threshold)
    return {
        "trial_id": trial_id,
        "threshold": threshold,
        "total": len(candidates),
        "candidates": candidates,
    }


@app.post("/recruitment/notify/{patient_id}/{trial_id}", tags=["Recruitment"], dependencies=[Depends(require_api_key)])
async def send_recruitment_notification(
    patient_id: str,
    trial_id: str,
    db: Session = Depends(get_db),
):
    """Send a recruitment email to a single patient for a given trial."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    from src.recruitment import RecruitmentEngine, _patient_to_dict
    features = EnrollmentPredictor._dict_to_features(_patient_to_dict(patient))
    ml_result = predictor.predict(features, patient_id, trial_id)

    candidate = {
        "patient_id": patient_id,
        "patient_name": f"{patient.first_name} {patient.last_name}",
        "email": patient.email or f"patient-{patient_id}@trial.local",
        "score": ml_result.enrollment_probability,
        "confidence": ml_result.confidence,
        "recommendation": ml_result.recommendation,
        "trial_id": trial_id,
        "trial_name": trial.name,
    }
    engine = RecruitmentEngine()
    email_sent = engine.send_recruitment_email(candidate)
    return {
        "notified": True,
        "patient_id": patient_id,
        "trial_id": trial_id,
        "email_sent": email_sent,
        "score": ml_result.enrollment_probability,
        "recommendation": ml_result.recommendation,
    }


# ------------------------------------------------------------------
# Admin — seed endpoint
# ------------------------------------------------------------------


@app.post("/admin/seed", tags=["Admin"], dependencies=[Depends(require_api_key)])
def seed_database(db: Session = Depends(get_db)):
    """Populate the database with synthetic patients, trials, and matches."""
    from src.seed_data import run_seed
    stats = run_seed(db)
    return {"seeded": True, "stats": stats}
