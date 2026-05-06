"""Database Models for Clinical Trial Cohort Matching System."""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


def _now() -> datetime:
    """Return the current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class Patient(Base):
    """ORM model representing a patient in the clinical trial system.

    Attributes:
        id: Unique patient identifier (string, max 50 chars).
        first_name: Patient given name.
        last_name: Patient family name.
        date_of_birth: Date/time of birth (timezone-aware).
        gender: Patient gender string.
        email: Contact email address (indexed for lookup).
        phone: Contact phone number.
        postal_code: Postal/ZIP code for proximity calculations.
        conditions: JSON list of ICD-10 coded condition dicts.
        medications: JSON list of ATC coded medication dicts.
        allergies: JSON list of allergy dicts.
        created_at: Record creation timestamp.
        updated_at: Record last-update timestamp.
        matches: Related PatientTrialMatch records.
    """

    __tablename__ = "patients"

    id: str = Column(String(50), primary_key=True, index=True)
    first_name: str = Column(String(100))
    last_name: str = Column(String(100))
    date_of_birth: Optional[datetime] = Column(DateTime(timezone=True))
    gender: str = Column(String(20))
    email: Optional[str] = Column(String(255), index=True)
    phone: Optional[str] = Column(String(20))
    postal_code: Optional[str] = Column(String(10))

    conditions = Column(JSON, default=list)
    medications = Column(JSON, default=list)
    allergies = Column(JSON, default=list)

    created_at: datetime = Column(DateTime(timezone=True), default=_now)
    updated_at: datetime = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    matches = relationship("PatientTrialMatch", back_populates="patient", cascade="all, delete-orphan")


class Trial(Base):
    """ORM model representing a clinical trial.

    Attributes:
        id: Unique trial identifier (string, max 50 chars).
        name: Trial display name.
        description: Free-text trial description.
        sponsor: Sponsoring organisation name.
        phase: Trial phase (Phase 1–4 or N/A).
        primary_condition: Primary condition under investigation.
        target_enrollment: Target participant count.
        status: Recruitment status (default RECRUITING).
        inclusion_criteria: JSON list of inclusion criterion dicts.
        exclusion_criteria: JSON list of exclusion criterion dicts.
        start_date: Trial start date.
        completion_date: Expected completion date.
        created_at: Record creation timestamp.
        updated_at: Record last-update timestamp.
        matches: Related PatientTrialMatch records.
    """

    __tablename__ = "trials"

    id: str = Column(String(50), primary_key=True, index=True)
    name: str = Column(String(255), nullable=False)
    description: Optional[str] = Column(Text)
    sponsor: Optional[str] = Column(String(255))
    phase: str = Column(String(10))
    primary_condition: str = Column(String(255))
    target_enrollment: int = Column(Integer)
    status: str = Column(String(20), default="RECRUITING")

    inclusion_criteria = Column(JSON, default=list)
    exclusion_criteria = Column(JSON, default=list)

    start_date: Optional[datetime] = Column(DateTime(timezone=True))
    completion_date: Optional[datetime] = Column(DateTime(timezone=True))

    created_at: datetime = Column(DateTime(timezone=True), default=_now)
    updated_at: datetime = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    matches = relationship("PatientTrialMatch", back_populates="trial", cascade="all, delete-orphan")


class PatientTrialMatch(Base):
    """ORM model representing a patient-trial eligibility match record.

    Stores both rule-based and ML-derived scores, match status, criteria details,
    and outreach tracking fields.
    """

    __tablename__ = "patient_trial_matches"

    id: int = Column(Integer, primary_key=True)
    patient_id: str = Column(String(50), ForeignKey("patients.id"), index=True)
    trial_id: str = Column(String(50), ForeignKey("trials.id"), index=True)

    rule_match_score: float = Column(Float, default=0.0)
    ml_match_score: float = Column(Float, default=0.0)
    enrollment_probability: float = Column(Float, default=0.0)
    combined_score: float = Column(Float, default=0.0)

    match_status: str = Column(String(20), default="PENDING")

    matched_criteria = Column(JSON, default=list)
    violated_criteria = Column(JSON, default=list)
    reasons = Column(JSON, default=list)

    letter_sent: bool = Column(Boolean, default=False)
    letter_sent_date: Optional[datetime] = Column(DateTime(timezone=True))
    enrolled: bool = Column(Boolean, default=False)
    enrollment_date: Optional[datetime] = Column(DateTime(timezone=True))

    created_at: datetime = Column(DateTime(timezone=True), default=_now)
    updated_at: datetime = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        UniqueConstraint("patient_id", "trial_id", name="uq_patient_trial"),
    )

    patient = relationship("Patient", back_populates="matches")
    trial = relationship("Trial", back_populates="matches")


DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://trialmatch:changeme@trial_postgres:5432/trial_db",
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    connect_args={} if DATABASE_URL.startswith("postgresql") else {"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all database tables if they do not exist.

    Raises:
        Exception: Re-raises any SQLAlchemy engine errors after logging.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created.")
    except Exception as exc:
        logger.error("Database init error: %s", exc)
        raise
