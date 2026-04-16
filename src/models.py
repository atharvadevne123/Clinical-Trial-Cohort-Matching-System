"""Database Models"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime,
    JSON, Boolean, ForeignKey, UniqueConstraint, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String(50), primary_key=True, index=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    date_of_birth = Column(DateTime(timezone=True))
    gender = Column(String(20))
    email = Column(String(255), index=True)
    phone = Column(String(20))
    postal_code = Column(String(10))

    conditions = Column(JSON, default=list)
    medications = Column(JSON, default=list)
    allergies = Column(JSON, default=list)

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    matches = relationship("PatientTrialMatch", back_populates="patient", cascade="all, delete-orphan")


class Trial(Base):
    __tablename__ = "trials"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    sponsor = Column(String(255))
    phase = Column(String(10))
    primary_condition = Column(String(255))
    target_enrollment = Column(Integer)
    status = Column(String(20), default="RECRUITING")

    inclusion_criteria = Column(JSON, default=list)
    exclusion_criteria = Column(JSON, default=list)

    start_date = Column(DateTime(timezone=True))
    completion_date = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    matches = relationship("PatientTrialMatch", back_populates="trial", cascade="all, delete-orphan")


class PatientTrialMatch(Base):
    __tablename__ = "patient_trial_matches"

    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), ForeignKey("patients.id"), index=True)
    trial_id = Column(String(50), ForeignKey("trials.id"), index=True)

    # rule_match_score: 0–100 from eligibility engine
    rule_match_score = Column(Float, default=0.0)
    # ml_match_score: 0–100 scaled ML score
    ml_match_score = Column(Float, default=0.0)
    # enrollment_probability: raw 0.0–1.0 from ML model
    enrollment_probability = Column(Float, default=0.0)
    # combined_score: weighted blend, 0–100
    combined_score = Column(Float, default=0.0)

    match_status = Column(String(20), default="PENDING")

    matched_criteria = Column(JSON, default=list)
    violated_criteria = Column(JSON, default=list)
    reasons = Column(JSON, default=list)

    letter_sent = Column(Boolean, default=False)
    letter_sent_date = Column(DateTime(timezone=True))
    enrolled = Column(Boolean, default=False)
    enrollment_date = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        UniqueConstraint("patient_id", "trial_id", name="uq_patient_trial"),
    )

    patient = relationship("Patient", back_populates="matches")
    trial = relationship("Trial", back_populates="matches")


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://trialmatch:changeme@trial_postgres:5432/trial_db"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    import logging
    logger = logging.getLogger(__name__)
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created.")
    except Exception as e:
        logger.error(f"Database init error: {e}")
        raise
