"""Step 4: Eligibility Matching Engine"""

from typing import Dict, Any
from datetime import datetime


class EligibilityMatcher:
    """Engine for matching patients to trials"""
    
    def __init__(self):
        self.operators = {
            "EQ": self._eq,
            "GT": self._gt,
            "LT": self._lt,
            "GTE": self._gte,
            "LTE": self._lte,
            "IN": self._in,
            "EXISTS": self._exists,
            "NOT_EXISTS": self._not_exists,
        }
    
    def check_match(self, patient: Dict[str, Any], trial: Dict[str, Any]) -> Dict[str, Any]:
        """Check if patient matches trial criteria"""
        matched_inclusion = []
        violated_exclusion = []
        reasons = []
        
        inclusion_criteria = trial.get("inclusion_criteria", [])
        if inclusion_criteria:
            for criterion in inclusion_criteria:
                if self._evaluate_criterion(patient, criterion):
                    matched_inclusion.append(criterion)
        
        exclusion_criteria = trial.get("exclusion_criteria", [])
        if exclusion_criteria:
            for criterion in exclusion_criteria:
                if self._evaluate_criterion(patient, criterion):
                    violated_exclusion.append(criterion)
        
        inclusion_score = len(matched_inclusion) / len(inclusion_criteria) * 100 if inclusion_criteria else 100
        exclusion_score = (1 - len(violated_exclusion) / len(exclusion_criteria)) * 100 if exclusion_criteria else 100
        
        match_score = int((inclusion_score * 0.7) + (exclusion_score * 0.3))
        
        is_eligible = (
            len(matched_inclusion) >= len(inclusion_criteria) // 2 if inclusion_criteria else True
        ) and len(violated_exclusion) == 0
        
        if matched_inclusion:
            reasons.append(f"Meets {len(matched_inclusion)}/{len(inclusion_criteria)} inclusion criteria")
        if violated_exclusion:
            reasons.append(f"Violates {len(violated_exclusion)} exclusion criteria")
            is_eligible = False
        
        if match_score >= 80:
            reasons.append("Strong match")
        elif match_score >= 60:
            reasons.append("Moderate match")
        else:
            reasons.append("Weak match")
        
        return {
            "eligible": is_eligible,
            "match_score": match_score,
            "matched_inclusion": matched_inclusion,
            "violated_exclusion": violated_exclusion,
            "reasons": reasons
        }
    
    def _evaluate_criterion(self, patient: Dict[str, Any], criterion: Dict[str, Any]) -> bool:
        """Evaluate a single criterion"""
        field = criterion.get("field")
        operator = criterion.get("operator", "EXISTS")
        value = criterion.get("value")
        
        patient_value = self._get_patient_field(patient, field)
        
        if operator not in self.operators:
            return False
        
        return self.operators[operator](patient_value, value)
    
    def _get_patient_field(self, patient: Dict[str, Any], field: str) -> Any:
        """Extract field value from patient data"""
        if "." in field:
            parts = field.split(".")
            value = patient
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value
        
        if field == "age":
            dob = patient.get("date_of_birth")
            if dob:
                if isinstance(dob, str):
                    dob = datetime.fromisoformat(dob.replace("Z", "+00:00"))
                age = (datetime.now(dob.tzinfo) - dob).days // 365
                return age
            return None
        
        if field.startswith("condition:"):
            code = field.replace("condition:", "")
            conditions = patient.get("conditions", [])
            for condition in conditions:
                if condition.get("code") == code or condition.get("icd10_code") == code:
                    return code
            return None
        
        if field.startswith("medication:"):
            code = field.replace("medication:", "")
            medications = patient.get("medications", [])
            for med in medications:
                if med.get("code") == code or med.get("medication_code") == code:
                    return code
            return None
        
        return patient.get(field)
    
    def _eq(self, a: Any, b: Any) -> bool:
        return a == b
    
    def _gt(self, a: Any, b: Any) -> bool:
        try:
            return float(a or 0) > float(b or 0)
        except:
            return False
    
    def _lt(self, a: Any, b: Any) -> bool:
        try:
            return float(a or 0) < float(b or 0)
        except:
            return False
    
    def _gte(self, a: Any, b: Any) -> bool:
        try:
            return float(a or 0) >= float(b or 0)
        except:
            return False
    
    def _lte(self, a: Any, b: Any) -> bool:
        try:
            return float(a or 0) <= float(b or 0)
        except:
            return False
    
    def _in(self, a: Any, b: Any) -> bool:
        if isinstance(a, list):
            return any(item in str(b).split(",") for item in a)
        return str(a) in str(b).split(",")
    
    def _exists(self, a: Any, b: Any) -> bool:
        return a is not None
    
    def _not_exists(self, a: Any, b: Any) -> bool:
        return a is None


matcher = EligibilityMatcher()
