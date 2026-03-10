"""Step 3: Pydantic Schemas"""

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class PatientCreate(BaseModel):
    id: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    gender: str
    email: Optional[str] = None
    phone: Optional[str] = None
    postal_code: Optional[str] = None
    conditions: Optional[List[dict]] = []
    medications: Optional[List[dict]] = []

class PatientResponse(BaseModel):
    id: str
    first_name: str
    last_name: str
    gender: str
    email: Optional[str]
    phone: Optional[str]
    conditions: Optional[List[dict]]
    medications: Optional[List[dict]]
    created_at: datetime
    
    class Config:
        from_attributes = True

class TrialCreate(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    sponsor: Optional[str] = None
    phase: str
    primary_condition: str
    target_enrollment: int
    inclusion_criteria: Optional[List[dict]] = []
    exclusion_criteria: Optional[List[dict]] = []
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None

class TrialResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    phase: str
    primary_condition: str
    target_enrollment: int
    status: str
    inclusion_criteria: Optional[List[dict]]
    exclusion_criteria: Optional[List[dict]]
    created_at: datetime
    
    class Config:
        from_attributes = True

class MatchResponse(BaseModel):
    id: int
    patient_id: str
    trial_id: str
    rule_match_score: int
    ml_match_score: int
    enrollment_probability: int
    combined_score: int
    match_status: str
    matched_criteria: Optional[List[dict]]
    violated_criteria: Optional[List[dict]]
    created_at: datetime
    
    class Config:
        from_attributes = True
