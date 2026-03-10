"""Step 5: FastAPI with NLP Entity Extraction"""

import logging
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from src.models import init_db, Patient, Trial, PatientTrialMatch, SessionLocal
from src.schemas import PatientCreate, PatientResponse, TrialCreate, TrialResponse
from src.eligibility import matcher
from src.nlp import nlp_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clinical Trial Cohort Matching - Step 5")

# Request schemas for NLP
class ClinicalNoteRequest(BaseModel):
    text: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup():
    logger.info("🚀 Initializing database...")
    try:
        init_db()
        logger.info("✅ Database tables created successfully!")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")

@app.get("/")
def read_root():
    return {"message": "Step 5: API with NLP Entity Extraction!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/status")
def status():
    return {"api": "running", "database": "connected", "step": "5 - NLP Entity Extraction"}

# ============== Patient Endpoints ==============

@app.post("/patients", response_model=PatientResponse)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient"""
    db_patient = db.query(Patient).filter(Patient.id == patient.id).first()
    if db_patient:
        raise HTTPException(status_code=400, detail="Patient already exists")
    
    db_patient = Patient(
        id=patient.id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        date_of_birth=patient.date_of_birth,
        gender=patient.gender,
        email=patient.email,
        phone=patient.phone,
        postal_code=patient.postal_code,
        conditions=patient.conditions,
        medications=patient.medications
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    logger.info(f"✅ Created patient: {patient.id}")
    return db_patient

@app.get("/patients/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """Get a patient by ID"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@app.get("/patients", response_model=list[PatientResponse])
def list_patients(db: Session = Depends(get_db)):
    """List all patients"""
    patients = db.query(Patient).all()
    return patients

# ============== Trial Endpoints ==============

@app.post("/trials", response_model=TrialResponse)
def create_trial(trial: TrialCreate, db: Session = Depends(get_db)):
    """Create a new trial"""
    db_trial = db.query(Trial).filter(Trial.id == trial.id).first()
    if db_trial:
        raise HTTPException(status_code=400, detail="Trial already exists")
    
    db_trial = Trial(
        id=trial.id,
        name=trial.name,
        description=trial.description,
        sponsor=trial.sponsor,
        phase=trial.phase,
        primary_condition=trial.primary_condition,
        target_enrollment=trial.target_enrollment,
        inclusion_criteria=trial.inclusion_criteria,
        exclusion_criteria=trial.exclusion_criteria,
        start_date=trial.start_date,
        completion_date=trial.completion_date
    )
    db.add(db_trial)
    db.commit()
    db.refresh(db_trial)
    logger.info(f"✅ Created trial: {trial.id}")
    return db_trial

@app.get("/trials/{trial_id}", response_model=TrialResponse)
def get_trial(trial_id: str, db: Session = Depends(get_db)):
    """Get a trial by ID"""
    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    return trial

@app.get("/trials", response_model=list[TrialResponse])
def list_trials(db: Session = Depends(get_db)):
    """List all trials"""
    trials = db.query(Trial).all()
    return trials

# ============== Eligibility Matching Endpoints ==============

@app.post("/match/{patient_id}/{trial_id}")
def check_match(patient_id: str, trial_id: str, db: Session = Depends(get_db)):
    """Check if a patient is eligible for a trial"""
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
        "conditions": patient.conditions,
        "medications": patient.medications
    }
    
    trial_dict = {
        "id": trial.id,
        "inclusion_criteria": trial.inclusion_criteria,
        "exclusion_criteria": trial.exclusion_criteria
    }
    
    result = matcher.check_match(patient_dict, trial_dict)
    
    match = PatientTrialMatch(
        patient_id=patient_id,
        trial_id=trial_id,
        rule_match_score=result["match_score"],
        match_status="ELIGIBLE" if result["eligible"] else "INELIGIBLE",
        matched_criteria=result["matched_inclusion"],
        violated_criteria=result["violated_exclusion"],
        reasons=result["reasons"]
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    
    logger.info(f"✅ Match result: {patient_id} -> {trial_id}: {result['eligible']}")
    
    return {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "eligible": result["eligible"],
        "match_score": result["match_score"],
        "matched_criteria": result["matched_inclusion"],
        "violated_criteria": result["violated_exclusion"],
        "reasons": result["reasons"]
    }

@app.get("/matches/{trial_id}")
def get_trial_matches(trial_id: str, db: Session = Depends(get_db)):
    """Get all matches for a trial"""
    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    
    matches = db.query(PatientTrialMatch).filter(PatientTrialMatch.trial_id == trial_id).all()
    
    return {
        "trial_id": trial_id,
        "total_matches": len(matches),
        "eligible_count": len([m for m in matches if m.match_status == "ELIGIBLE"]),
        "matches": matches
    }

# ============== NLP Endpoints ==============

@app.post("/nlp/extract-entities")
def extract_clinical_entities(request: ClinicalNoteRequest):
    """Extract medical entities from clinical text"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    entities = nlp_processor.extract_entities(request.text)
    
    return {
        "text_length": len(request.text),
        "entities_found": len(entities["conditions"]) + len(entities["medications"]) + len(entities["symptoms"]),
        "extraction_result": entities
    }

@app.post("/nlp/clinical-profile")
def generate_clinical_profile(request: ClinicalNoteRequest):
    """Generate a clinical profile summary from text"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    profile = nlp_processor.summarize_clinical_profile(request.text)
    
    return {
        "clinical_profile": profile,
        "text_analyzed": len(request.text)
    }

@app.post("/patients/{patient_id}/analyze-notes")
def analyze_patient_notes(patient_id: str, request: ClinicalNoteRequest, db: Session = Depends(get_db)):
    """Analyze clinical notes and update patient conditions/medications"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Extract entities from notes
    entities = nlp_processor.extract_entities(request.text)
    
    # Update patient data with extracted entities
    if entities["conditions"]:
        patient.conditions = entities["conditions"]
    
    if entities["medications"]:
        patient.medications = entities["medications"]
    
    db.commit()
    db.refresh(patient)
    
    logger.info(f"✅ Analyzed clinical notes for patient {patient_id}")
    
    return {
        "patient_id": patient_id,
        "extracted_conditions": entities["conditions"],
        "extracted_medications": entities["medications"],
        "extracted_symptoms": entities["symptoms"],
        "patient_updated": True
    }
from .ml_prediction import create_ml_router; app.include_router(create_ml_router())
