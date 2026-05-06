"""Pydantic Schemas for Clinical Trial Cohort Matching API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PatientCreate(BaseModel):
    """Schema for creating a new patient record.

    Attributes:
        id: Unique patient identifier (max 50 chars).
        first_name: Patient given name.
        last_name: Patient family name.
        date_of_birth: Date and time of birth (timezone-aware recommended).
        gender: Patient gender (male, female, other, unknown).
        email: Optional contact email address.
        phone: Optional contact phone number.
        postal_code: Optional postal/ZIP code.
        conditions: List of condition dicts (code, name).
        medications: List of medication dicts (code, name).
        allergies: List of allergy dicts.
    """

    id: str = Field(..., min_length=1, max_length=50)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: datetime
    gender: str = Field(..., min_length=1, max_length=20)
    email: Optional[str] = Field(default=None, max_length=255)
    phone: Optional[str] = Field(default=None, max_length=20)
    postal_code: Optional[str] = Field(default=None, max_length=10)
    conditions: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    medications: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    allergies: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    @field_validator("gender")
    @classmethod
    def gender_must_be_valid(cls, v: str) -> str:
        """Normalise gender to lowercase and validate allowed values."""
        normalised = v.strip().lower()
        allowed = {"male", "female", "other", "unknown"}
        if normalised not in allowed:
            raise ValueError(f"gender must be one of {sorted(allowed)}")
        return normalised

    @field_validator("email")
    @classmethod
    def email_basic_format(cls, v: Optional[str]) -> Optional[str]:
        """Ensure email contains an @ sign if provided."""
        if v and "@" not in v:
            raise ValueError("email must contain @")
        return v


class PatientResponse(BaseModel):
    """Schema for patient records returned by the API."""

    id: str
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    gender: str
    email: Optional[str] = None
    phone: Optional[str] = None
    postal_code: Optional[str] = None
    conditions: Optional[List[Dict[str, Any]]] = None
    medications: Optional[List[Dict[str, Any]]] = None
    allergies: Optional[List[Dict[str, Any]]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TrialCreate(BaseModel):
    """Schema for creating a new clinical trial record.

    Attributes:
        id: Unique trial identifier (max 50 chars).
        name: Trial display name.
        description: Optional trial description.
        sponsor: Optional sponsoring organisation.
        phase: Trial phase (e.g. Phase 1, Phase 2).
        primary_condition: Primary condition under study.
        target_enrollment: Target number of participants.
        inclusion_criteria: List of inclusion criterion dicts.
        exclusion_criteria: List of exclusion criterion dicts.
        start_date: Optional trial start date.
        completion_date: Optional trial completion date.
    """

    id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    sponsor: Optional[str] = Field(default=None, max_length=255)
    phase: str = Field(..., min_length=1, max_length=10)
    primary_condition: str = Field(..., min_length=1, max_length=255)
    target_enrollment: int = Field(..., gt=0)
    inclusion_criteria: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    exclusion_criteria: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None

    @field_validator("phase")
    @classmethod
    def phase_must_be_valid(cls, v: str) -> str:
        """Validate that the trial phase is a recognised value."""
        allowed = {"Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 1/2", "Phase 2/3", "N/A"}
        if v not in allowed:
            raise ValueError(f"phase must be one of {sorted(allowed)}")
        return v


class TrialResponse(BaseModel):
    """Schema for trial records returned by the API."""

    id: str
    name: str
    description: Optional[str] = None
    sponsor: Optional[str] = None
    phase: str
    primary_condition: str
    target_enrollment: int
    status: str
    inclusion_criteria: Optional[List[Dict[str, Any]]] = None
    exclusion_criteria: Optional[List[Dict[str, Any]]] = None
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MatchResponse(BaseModel):
    """Schema for patient-trial match records returned by the API."""

    id: int
    patient_id: str
    trial_id: str
    rule_match_score: float
    ml_match_score: float
    enrollment_probability: float
    combined_score: float
    match_status: str
    matched_criteria: Optional[List[Dict[str, Any]]] = None
    violated_criteria: Optional[List[Dict[str, Any]]] = None
    reasons: Optional[List[str]] = None
    letter_sent: bool
    enrolled: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ClinicalNoteRequest(BaseModel):
    """Schema for NLP clinical note analysis requests."""

    text: str = Field(..., min_length=1, description="Clinical note text to analyse")
