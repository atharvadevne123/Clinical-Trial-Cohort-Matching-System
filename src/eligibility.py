"""Eligibility Matching Engine"""

from typing import Dict, Any
from datetime import datetime


class EligibilityMatcher:
    """Engine for matching patients to clinical trial criteria."""

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
        """Check if a patient meets all trial criteria."""
        matched_inclusion = []
        violated_exclusion = []
        reasons = []

        inclusion_criteria = trial.get("inclusion_criteria", [])
        for criterion in inclusion_criteria:
            if self._evaluate_criterion(patient, criterion):
                matched_inclusion.append(criterion)

        exclusion_criteria = trial.get("exclusion_criteria", [])
        for criterion in exclusion_criteria:
            if self._evaluate_criterion(patient, criterion):
                violated_exclusion.append(criterion)

        # Inclusion score: fraction of criteria met (0–100)
        inclusion_score = (
            len(matched_inclusion) / len(inclusion_criteria) * 100
            if inclusion_criteria else 100
        )
        # Exclusion score: penalise any violated criteria
        exclusion_score = (
            (1 - len(violated_exclusion) / len(exclusion_criteria)) * 100
            if exclusion_criteria else 100
        )

        match_score = round((inclusion_score * 0.7) + (exclusion_score * 0.3), 1)

        # ALL inclusion criteria must be met; NO exclusion criteria may be triggered
        is_eligible = (
            len(matched_inclusion) == len(inclusion_criteria)
        ) and len(violated_exclusion) == 0

        if inclusion_criteria:
            reasons.append(
                f"Meets {len(matched_inclusion)}/{len(inclusion_criteria)} inclusion criteria"
            )
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
            "reasons": reasons,
        }

    def _evaluate_criterion(self, patient: Dict[str, Any], criterion: Dict[str, Any]) -> bool:
        field = criterion.get("field")
        operator = criterion.get("operator", "EXISTS")
        value = criterion.get("value")

        if not field or operator not in self.operators:
            return False

        patient_value = self._get_patient_field(patient, field)
        return self.operators[operator](patient_value, value)

    def _get_patient_field(self, patient: Dict[str, Any], field: str) -> Any:
        # Check prefixed fields FIRST — before the dot check, because ICD-10
        # codes like "I48.91" contain dots which would otherwise trigger the
        # nested-path branch and silently return None.
        if field == "age":
            dob = patient.get("date_of_birth")
            if dob:
                if isinstance(dob, str):
                    dob = datetime.fromisoformat(dob.replace("Z", "+00:00"))
                now = datetime.now(dob.tzinfo) if dob.tzinfo else datetime.now()
                return (now - dob).days // 365
            return None

        if field.startswith("condition:"):
            code = field[len("condition:"):]
            for cond in patient.get("conditions", []) or []:
                if isinstance(cond, dict):
                    if cond.get("code") == code or cond.get("icd10_code") == code:
                        return code
                elif str(cond) == code:
                    return code
            return None

        if field.startswith("medication:"):
            code = field[len("medication:"):]
            for med in patient.get("medications", []) or []:
                if isinstance(med, dict):
                    if med.get("code") == code or med.get("medication_code") == code:
                        return code
                elif str(med) == code:
                    return code
            return None

        # Generic nested-path lookup (e.g. "address.city") — safe now that
        # condition:/medication: prefixes are already handled above.
        if "." in field:
            value = patient
            for part in field.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value

        return patient.get(field)

    # --- comparison helpers ---

    def _eq(self, a: Any, b: Any) -> bool:
        return a == b

    def _gt(self, a: Any, b: Any) -> bool:
        try:
            return float(a) > float(b)
        except (TypeError, ValueError):
            return False

    def _lt(self, a: Any, b: Any) -> bool:
        try:
            return float(a) < float(b)
        except (TypeError, ValueError):
            return False

    def _gte(self, a: Any, b: Any) -> bool:
        try:
            return float(a) >= float(b)
        except (TypeError, ValueError):
            return False

    def _lte(self, a: Any, b: Any) -> bool:
        try:
            return float(a) <= float(b)
        except (TypeError, ValueError):
            return False

    def _in(self, a: Any, b: Any) -> bool:
        options = [s.strip() for s in str(b).split(",")]
        if isinstance(a, list):
            return any(str(item) in options for item in a)
        return str(a) in options

    def _exists(self, a: Any, b: Any) -> bool:
        return a is not None

    def _not_exists(self, a: Any, b: Any) -> bool:
        return a is None


matcher = EligibilityMatcher()
