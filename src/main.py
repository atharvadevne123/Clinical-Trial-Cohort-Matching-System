"""Clinical Trial Cohort Matching API.

FastAPI application providing patient-trial matching via rule-based eligibility
evaluation and XGBoost ML enrollment prediction with NLP and FHIR integration.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from src.eligibility import matcher
from src.fhir import fhir_client
from src.ml_prediction import EnrollmentPredictor, create_ml_router, predictor
from src.models import Patient, PatientTrialMatch, SessionLocal, Trial, init_db
from src.nlp import nlp_processor
from src.schemas import (
    ClinicalNoteRequest,
    PatientCreate,
    PatientResponse,
    TrialCreate,
    TrialResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

API_KEY: str = os.environ.get("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_START_TIME: datetime = datetime.now(timezone.utc)

if not API_KEY:
    logger.warning(
        "API_KEY env var is not set — all endpoints are unauthenticated. "
        "Set API_KEY in production."
    )


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> Optional[str]:
    """Validate the X-API-Key header when API_KEY env var is set.

    Args:
        key: Value from the X-API-Key header, or None if absent.

    Returns:
        The validated key, or None in dev mode (API_KEY not set).

    Raises:
        HTTPException: 403 if API_KEY is set and the header does not match.
    """
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")
    return key


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the database on startup."""
    logger.info("Initialising database...")
    try:
        init_db()
        logger.info("Database ready.")
    except Exception as exc:
        logger.error("Database init failed: %s", exc)
        raise
    yield


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Clinical Trial Cohort Matching",
    description=(
        "Match patients to clinical trials using rule-based eligibility and "
        "XGBoost ML enrollment prediction."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

_CORS_ORIGINS: list[str] = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
] or ["*"]

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next: Any) -> Response:
    """Inject X-Correlation-ID and X-Process-Time headers into every response."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    logger.debug("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
    return response


app.include_router(create_ml_router())


# ------------------------------------------------------------------
# Error handlers
# ------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured JSON body for 404 Not Found responses."""
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "detail": str(getattr(exc, "detail", "Resource not found")), "path": request.url.path},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a structured JSON body for 422 validation errors."""
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors(), "path": request.url.path},
    )


# ------------------------------------------------------------------
# DB dependency
# ------------------------------------------------------------------

def get_db() -> Session:
    """Yield a database session and ensure it is closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------
# Health / meta
# ------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root() -> Dict[str, str]:
    """Return API name and version metadata."""
    return {"message": "Clinical Trial Cohort Matching API", "version": "1.1.0", "docs": "/docs"}


@app.get("/ping", tags=["Meta"])
def ping() -> Dict[str, str]:
    """Ultra-lightweight liveness probe that returns pong immediately."""
    return {"ping": "pong"}


@app.get("/health", tags=["Meta"])
def health_check() -> Dict[str, str]:
    """Liveness probe — always returns healthy when the process is running."""
    return {"status": "healthy"}


@app.get("/healthz", tags=["Meta"])
def healthz() -> Dict[str, str]:
    """Kubernetes-style liveness probe endpoint."""
    return {"status": "ok"}


@app.get("/version", tags=["Meta"])
def version() -> Dict[str, Any]:
    """Return API version and build metadata."""
    return {
        "version": "1.1.0",
        "api": "Clinical Trial Cohort Matching",
        "python_version": __import__("sys").version,
        "started_at": _START_TIME.isoformat(),
    }


@app.get("/status", tags=["Meta"])
def status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return live record counts for patients, trials, and matches."""
    return {
        "api": "running",
        "patients": db.query(Patient).count(),
        "trials": db.query(Trial).count(),
        "matches": db.query(PatientTrialMatch).count(),
    }


@app.get("/metrics", tags=["Meta"])
def metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return runtime statistics including record counts and uptime."""
    uptime_seconds = (datetime.now(timezone.utc) - _START_TIME).total_seconds()
    eligible = db.query(PatientTrialMatch).filter(
        PatientTrialMatch.match_status == "ELIGIBLE"
    ).count()
    enrolled = db.query(PatientTrialMatch).filter(
        PatientTrialMatch.enrolled == True  # noqa: E712
    ).count()
    return {
        "uptime_seconds": round(uptime_seconds, 1),
        "patients": db.query(Patient).count(),
        "trials": db.query(Trial).count(),
        "total_matches": db.query(PatientTrialMatch).count(),
        "eligible_matches": eligible,
        "enrolled_patients": enrolled,
    }


# ------------------------------------------------------------------
# Patients
# ------------------------------------------------------------------

@app.post("/patients", response_model=PatientResponse, tags=["Patients"],
          dependencies=[Depends(require_api_key)])
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)) -> Patient:
    """Create a new patient record.

    Args:
        patient: Validated PatientCreate payload.
        db: Database session (injected).

    Raises:
        HTTPException: 409 if a patient with the same ID already exists.
    """
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
    logger.info("Created patient %s", db_patient.id)
    return db_patient


@app.get("/patients/{patient_id}", response_model=PatientResponse, tags=["Patients"])
def get_patient(patient_id: str, db: Session = Depends(get_db)) -> Patient:
    """Retrieve a patient record by ID.

    Raises:
        HTTPException: 404 if the patient is not found.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/patients", response_model=List[PatientResponse], tags=["Patients"])
def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    gender: Optional[str] = Query(None, description="Filter by gender (male/female/other)"),
    condition: Optional[str] = Query(None, description="Filter by ICD-10 condition code"),
    db: Session = Depends(get_db),
) -> List[Patient]:
    """List patients with optional gender and condition filters."""
    q = db.query(Patient)
    if gender:
        q = q.filter(Patient.gender == gender.lower())
    if condition:
        q = q.filter(Patient.conditions.contains([{"code": condition}]))
    return q.offset(skip).limit(limit).all()


# ------------------------------------------------------------------
# Trials
# ------------------------------------------------------------------

@app.post("/trials", response_model=TrialResponse, tags=["Trials"],
          dependencies=[Depends(require_api_key)])
def create_trial(trial: TrialCreate, db: Session = Depends(get_db)) -> Trial:
    """Create a new clinical trial record.

    Raises:
        HTTPException: 409 if a trial with the same ID already exists.
    """
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
    logger.info("Created trial %s", db_trial.id)
    return db_trial


@app.get("/trials/{trial_id}", response_model=TrialResponse, tags=["Trials"])
def get_trial(trial_id: str, db: Session = Depends(get_db)) -> Trial:
    """Retrieve a trial record by ID.

    Raises:
        HTTPException: 404 if the trial is not found.
    """
    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    return trial


@app.get("/trials", response_model=List[TrialResponse], tags=["Trials"])
def list_trials(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    phase: Optional[str] = Query(None, description="Filter by trial phase (e.g. 'Phase 2')"),
    status: Optional[str] = Query(None, description="Filter by recruitment status"),
    db: Session = Depends(get_db),
) -> List[Trial]:
    """List trials with optional phase and status filters."""
    q = db.query(Trial)
    if phase:
        q = q.filter(Trial.phase == phase)
    if status:
        q = q.filter(Trial.recruitment_status == status)
    return q.offset(skip).limit(limit).all()


# ------------------------------------------------------------------
# Eligibility matching (rule-based + ML)
# ------------------------------------------------------------------

_COMBINED_RULE_WEIGHT: float = 0.5
_COMBINED_ML_WEIGHT: float = 0.5


def _compute_combined_score(rule_score: float, ml_score: float) -> float:
    """Compute the weighted blend of rule-based and ML match scores.

    Args:
        rule_score: Rule-based eligibility score (0–100).
        ml_score: ML enrollment probability score (0–100).

    Returns:
        Combined score (0–100) as 50% rule + 50% ML.
    """
    return round(_COMBINED_RULE_WEIGHT * rule_score + _COMBINED_ML_WEIGHT * ml_score, 2)


@app.post("/match/{patient_id}/{trial_id}", tags=["Matching"],
          dependencies=[Depends(require_api_key)])
def check_match(
    patient_id: str, trial_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Run rule-based and ML matching for a patient-trial pair.

    Creates a PatientTrialMatch record and returns the combined result.

    Raises:
        HTTPException: 409 if a match record already exists.
        HTTPException: 404 if patient or trial is not found.
    """
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

    patient_dict: Dict[str, Any] = {
        "id": patient.id,
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "date_of_birth": patient.date_of_birth,
        "gender": patient.gender,
        "conditions": patient.conditions or [],
        "medications": patient.medications or [],
    }
    trial_dict: Dict[str, Any] = {
        "id": trial.id,
        "inclusion_criteria": trial.inclusion_criteria or [],
        "exclusion_criteria": trial.exclusion_criteria or [],
    }

    rule_result = matcher.check_match(patient_dict, trial_dict)
    features = EnrollmentPredictor._dict_to_features(patient_dict)
    ml_result = predictor.predict(features, patient_id, trial_id)

    rule_score = float(rule_result["match_score"])
    ml_score = ml_result.enrollment_probability * 100
    combined = _compute_combined_score(rule_score, ml_score)

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
) -> Dict[str, Any]:
    """Return all match records for a trial with optional status filter."""
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
def get_patient_matches(patient_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return all match records for a patient, sorted by combined score descending."""
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

@app.post("/nlp/extract-entities", tags=["NLP"])
def extract_clinical_entities(request: ClinicalNoteRequest) -> Dict[str, Any]:
    """Extract clinical entities (conditions, medications, symptoms) from free text."""
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
def generate_clinical_profile(request: ClinicalNoteRequest) -> Dict[str, Any]:
    """Generate a structured clinical profile summary from free text."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return {
        "clinical_profile": nlp_processor.summarize_clinical_profile(request.text),
        "text_length": len(request.text),
    }


@app.post("/patients/{patient_id}/analyze-notes", tags=["NLP"],
          dependencies=[Depends(require_api_key)])
def analyze_patient_notes(
    patient_id: str, request: ClinicalNoteRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Extract clinical entities from notes and update the patient record."""
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

@app.post("/fhir/import/{fhir_patient_id}", tags=["FHIR"],
          dependencies=[Depends(require_api_key)])
def import_fhir_patient(
    fhir_patient_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Fetch a patient from the FHIR server and upsert into the local database."""
    profile = fhir_client.fetch_complete_patient_profile(fhir_patient_id)

    patient = db.query(Patient).filter(Patient.id == fhir_patient_id).first()
    if patient:
        patient.conditions = profile.get("conditions", [])
        patient.medications = profile.get("medications", [])
    else:
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
    logger.info("FHIR import complete for patient %s", fhir_patient_id)
    return {"imported": True, "patient_id": patient.id, "profile": profile}


# ------------------------------------------------------------------
# Recruitment
# ------------------------------------------------------------------

@app.get("/recruitment/candidates/{trial_id}", tags=["Recruitment"])
async def get_recruitment_candidates(
    trial_id: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum enrollment probability"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
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


@app.post("/recruitment/notify/{patient_id}/{trial_id}", tags=["Recruitment"],
          dependencies=[Depends(require_api_key)])
async def send_recruitment_notification(
    patient_id: str,
    trial_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
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
def seed_database(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Populate the database with synthetic patients, trials, and matches."""
    from src.seed_data import run_seed
    stats = run_seed(db)
    return {"seeded": True, "stats": stats}
