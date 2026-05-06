"""API v1 Router — versioned endpoint prefix for the matching API.

All core endpoints are re-exported under /api/v1/ to support API versioning.
The unversioned paths remain available for backwards compatibility.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.models import Patient, PatientTrialMatch, SessionLocal, Trial
from src.schemas import PatientCreate, PatientResponse, TrialCreate, TrialResponse
from src.main import get_db, require_api_key
from src.eligibility import matcher
from src.ml_prediction import EnrollmentPredictor, predictor

logger = logging.getLogger(__name__)

v1_router = APIRouter(prefix="/api/v1", tags=["API v1"])


@v1_router.get("/health", summary="Versioned liveness probe")
def v1_health() -> Dict[str, str]:
    """Return healthy status for the v1 API."""
    return {"status": "healthy", "version": "v1"}


@v1_router.get("/patients", response_model=List[PatientResponse], summary="List patients (v1)")
def v1_list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> List[Patient]:
    """List patients with pagination (API v1)."""
    return db.query(Patient).offset(skip).limit(limit).all()


@v1_router.get("/patients/{patient_id}", response_model=PatientResponse, summary="Get patient (v1)")
def v1_get_patient(patient_id: str, db: Session = Depends(get_db)) -> Patient:
    """Retrieve a patient by ID (API v1).

    Raises:
        HTTPException: 404 if not found.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@v1_router.get("/trials", response_model=List[TrialResponse], summary="List trials (v1)")
def v1_list_trials(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> List[Trial]:
    """List trials with pagination (API v1)."""
    return db.query(Trial).offset(skip).limit(limit).all()


@v1_router.get("/trials/{trial_id}", response_model=TrialResponse, summary="Get trial (v1)")
def v1_get_trial(trial_id: str, db: Session = Depends(get_db)) -> Trial:
    """Retrieve a trial by ID (API v1).

    Raises:
        HTTPException: 404 if not found.
    """
    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    return trial


@v1_router.get("/status", summary="System status (v1)")
def v1_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return live record counts (API v1)."""
    return {
        "api_version": "v1",
        "patients": db.query(Patient).count(),
        "trials": db.query(Trial).count(),
        "matches": db.query(PatientTrialMatch).count(),
    }
