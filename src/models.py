"""Step 2: Database Models"""

from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(String(50), primary_key=True, index=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    date_of_birth = Column(DateTime)
    gender = Column(String(20))
    email = Column(String(255), index=True)
    phone = Column(String(20))
    postal_code = Column(String(10))
    
    conditions = Column(JSON, default=[])
    medications = Column(JSON, default=[])
    allergies = Column(JSON, default=[])
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    matches = relationship("PatientTrialMatch", back_populates="patient", cascade="all, delete-orphan")

class Trial(Base):
    __tablename__ = "trials"
    
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(500))
    sponsor = Column(String(255))
    phase = Column(String(10))
    primary_condition = Column(String(255))
    target_enrollment = Column(Integer)
    status = Column(String(20), default="RECRUITING")
    
    inclusion_criteria = Column(JSON, default=[])
    exclusion_criteria = Column(JSON, default=[])
    
    start_date = Column(DateTime)
    completion_date = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    matches = relationship("PatientTrialMatch", back_populates="trial", cascade="all, delete-orphan")

class PatientTrialMatch(Base):
    __tablename__ = "patient_trial_matches"
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), ForeignKey("patients.id"), index=True)
    trial_id = Column(String(50), ForeignKey("trials.id"), index=True)
    
    rule_match_score = Column(Integer, default=0)
    ml_match_score = Column(Integer, default=0)
    enrollment_probability = Column(Integer, default=0)
    combined_score = Column(Integer, default=0)
    
    match_status = Column(String(20), default="PENDING")
    
    matched_criteria = Column(JSON, default=[])
    violated_criteria = Column(JSON, default=[])
    reasons = Column(JSON, default=[])
    
    letter_sent = Column(Boolean, default=False)
    letter_sent_date = Column(DateTime)
    enrolled = Column(Boolean, default=False)
    enrollment_date = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="matches")
    trial = relationship("Trial", back_populates="matches")

DATABASE_URL = "postgresql://trialmatch:changeme@postgres:5432/trialmatch"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    import logging
    logger = logging.getLogger(__name__)
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
