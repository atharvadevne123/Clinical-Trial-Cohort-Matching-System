"""Eligibility Matching Engine for Clinical Trial Cohort Matching."""

from datetime import datetime
from typing import Any, Dict, List, Optional

_INCLUSION_WEIGHT: float = 0.7
_EXCLUSION_WEIGHT: float = 0.3


class EligibilityMatcher:
    """Rule-based engine that evaluates patient eligibility against trial criteria.

    Supports operators: EQ, GT, LT, GTE, LTE, IN, EXISTS, NOT_EXISTS.
    Field prefixes ``condition:`` and ``medication:`` allow ICD-10/ATC code lookups.

    Score formula: (inclusion_pct × 0.7) + (exclusion_pct × 0.3)
    """

    def __init__(self) -> None:
        self.operators: Dict[str, Any] = {
            "EQ": self._eq,
            "GT": self._gt,
            "LT": self._lt,
            "GTE": self._gte,
            "LTE": self._lte,
            "IN": self._in,
            "NOT_IN": self._not_in,
            "CONTAINS": self._contains,
            "BETWEEN": self._between,
            "EXISTS": self._exists,
            "NOT_EXISTS": self._not_exists,
        }

    def check_match(self, patient: Dict[str, Any], trial: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all inclusion and exclusion criteria for a patient-trial pair.

        Args:
            patient: Patient record dict with id, conditions, medications, gender, etc.
            trial: Trial record dict with inclusion_criteria and exclusion_criteria lists.

        Returns:
            Dict with keys: eligible, match_score, matched_inclusion,
            violated_exclusion, reasons.
        """
        matched_inclusion: List[Dict[str, Any]] = []
        violated_exclusion: List[Dict[str, Any]] = []
        reasons: List[str] = []

        inclusion_criteria: List[Dict[str, Any]] = trial.get("inclusion_criteria", [])
        for criterion in inclusion_criteria:
            if self._evaluate_criterion(patient, criterion):
                matched_inclusion.append(criterion)

        exclusion_criteria: List[Dict[str, Any]] = trial.get("exclusion_criteria", [])
        for criterion in exclusion_criteria:
            if self._evaluate_criterion(patient, criterion):
                violated_exclusion.append(criterion)

        inclusion_score: float = (
            len(matched_inclusion) / len(inclusion_criteria) * 100 if inclusion_criteria else 100.0
        )
        exclusion_score: float = (
            (1 - len(violated_exclusion) / len(exclusion_criteria)) * 100
            if exclusion_criteria
            else 100.0
        )

        match_score: float = round(
            (inclusion_score * _INCLUSION_WEIGHT) + (exclusion_score * _EXCLUSION_WEIGHT), 1
        )

        is_eligible: bool = (len(matched_inclusion) == len(inclusion_criteria)) and len(
            violated_exclusion
        ) == 0

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
        """Evaluate a single criterion against a patient record.

        Args:
            patient: Patient record dict.
            criterion: Criterion dict with field, operator, and value keys.

        Returns:
            True if the criterion is satisfied.
        """
        field: Optional[str] = criterion.get("field")
        operator: str = criterion.get("operator", "EXISTS")
        value: Any = criterion.get("value")

        if not field or operator not in self.operators:
            return False

        patient_value = self._get_patient_field(patient, field)
        return self.operators[operator](patient_value, value)

    def _get_patient_field(self, patient: Dict[str, Any], field: str) -> Any:
        """Extract a field value from a patient record, supporting special prefixes.

        Handles ``age`` (computed from date_of_birth), ``condition:<code>``,
        ``medication:<code>``, and dot-separated nested paths.

        Args:
            patient: Patient record dict.
            field: Field name or prefixed field identifier.

        Returns:
            The extracted field value, or None if not found.
        """
        if field == "age":
            return self._calculate_age(patient)

        if field.startswith("condition:"):
            code = field[len("condition:") :]
            return self._find_condition_code(patient, code)

        if field.startswith("medication:"):
            code = field[len("medication:") :]
            return self._find_medication_code(patient, code)

        if "." in field:
            return self._nested_get(patient, field)

        return patient.get(field)

    def _calculate_age(self, patient: Dict[str, Any]) -> Optional[int]:
        """Compute patient age in years from date_of_birth.

        Args:
            patient: Patient record dict containing date_of_birth.

        Returns:
            Age in whole years, or None if date_of_birth is absent.
        """
        dob = patient.get("date_of_birth")
        if not dob:
            return None
        if isinstance(dob, str):
            dob = datetime.fromisoformat(dob.replace("Z", "+00:00"))
        now = datetime.now(dob.tzinfo) if dob.tzinfo else datetime.now()
        return (now - dob).days // 365

    def _find_condition_code(self, patient: Dict[str, Any], code: str) -> Optional[str]:
        """Look up an ICD-10 code in the patient's conditions list.

        Args:
            patient: Patient record dict.
            code: ICD-10 code to search for.

        Returns:
            The code string if found, otherwise None.
        """
        for cond in patient.get("conditions", []) or []:
            if isinstance(cond, dict):
                if cond.get("code") == code or cond.get("icd10_code") == code:
                    return code
            elif str(cond) == code:
                return code
        return None

    def _find_medication_code(self, patient: Dict[str, Any], code: str) -> Optional[str]:
        """Look up an ATC/medication code in the patient's medications list.

        Args:
            patient: Patient record dict.
            code: Medication code to search for.

        Returns:
            The code string if found, otherwise None.
        """
        for med in patient.get("medications", []) or []:
            if isinstance(med, dict):
                if med.get("code") == code or med.get("medication_code") == code:
                    return code
            elif str(med) == code:
                return code
        return None

    def _nested_get(self, obj: Dict[str, Any], path: str) -> Any:
        """Traverse a dot-separated path in a nested dict.

        Args:
            obj: Root dict to traverse.
            path: Dot-separated key path (e.g. "address.city").

        Returns:
            The value at the path, or None if any key is missing.
        """
        value: Any = obj
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    # --- comparison helpers ---

    def _eq(self, a: Any, b: Any) -> bool:
        """Return True if a equals b (exact match)."""
        return a == b

    def _gt(self, a: Any, b: Any) -> bool:
        """Return True if numeric a is strictly greater than numeric b."""
        try:
            return float(a) > float(b)
        except (TypeError, ValueError):
            return False

    def _lt(self, a: Any, b: Any) -> bool:
        """Return True if numeric a is strictly less than numeric b."""
        try:
            return float(a) < float(b)
        except (TypeError, ValueError):
            return False

    def _gte(self, a: Any, b: Any) -> bool:
        """Return True if numeric a is greater than or equal to numeric b."""
        try:
            return float(a) >= float(b)
        except (TypeError, ValueError):
            return False

    def _lte(self, a: Any, b: Any) -> bool:
        """Return True if numeric a is less than or equal to numeric b."""
        try:
            return float(a) <= float(b)
        except (TypeError, ValueError):
            return False

    def _in(self, a: Any, b: Any) -> bool:
        """Return True if a (or any element of a list a) appears in comma-separated b."""
        options = [s.strip() for s in str(b).split(",")]
        if isinstance(a, list):
            return any(str(item) in options for item in a)
        return str(a) in options

    def _not_in(self, a: Any, b: Any) -> bool:
        """Return True if a does not appear in comma-separated b."""
        options = [s.strip() for s in str(b).split(",")]
        if isinstance(a, list):
            return not any(str(item) in options for item in a)
        return str(a) not in options

    def _contains(self, a: Any, b: Any) -> bool:
        """Return True if string a contains substring b (case-insensitive)."""
        if a is None or b is None:
            return False
        return str(b).lower() in str(a).lower()

    def _between(self, a: Any, b: Any) -> bool:
        """Return True if numeric a is within the range [lo, hi] given as 'lo,hi'."""
        if a is None or b is None:
            return False
        try:
            parts = str(b).split(",")
            lo, hi = float(parts[0].strip()), float(parts[1].strip())
            return lo <= float(a) <= hi
        except (TypeError, ValueError, IndexError):
            return False

    def _exists(self, a: Any, b: Any) -> bool:
        """Return True if a is not None."""
        return a is not None

    def _not_exists(self, a: Any, b: Any) -> bool:
        """Return True if a is None."""
        return a is None

    @property
    def operator_names(self) -> List[str]:
        """Return a sorted list of registered operator names."""
        return sorted(self.operators.keys())

    def __repr__(self) -> str:
        return f"EligibilityMatcher(operators={self.operator_names})"

    def count_eligible(
        self, patients: List[Dict[str, Any]], trial: Dict[str, Any]
    ) -> int:
        """Count how many patients are eligible for a trial.

        Args:
            patients: List of patient record dicts.
            trial: Trial record dict with inclusion/exclusion criteria.

        Returns:
            Number of eligible patients.
        """
        return sum(1 for p in patients if self.check_match(p, trial)["eligible"])

    def score_candidates(
        self, patients: List[Dict[str, Any]], trial: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score a list of patients against a trial, sorted by match_score descending.

        Args:
            patients: List of patient record dicts.
            trial: Trial record dict with inclusion_criteria and exclusion_criteria.

        Returns:
            List of result dicts (from check_match) with patient_id added, sorted descending.
        """
        results = []
        for patient in patients:
            result = self.check_match(patient, trial)
            result["patient_id"] = patient.get("id", "unknown")
            results.append(result)
        return sorted(results, key=lambda r: r["match_score"], reverse=True)


matcher = EligibilityMatcher()

__all__ = [
    "EligibilityMatcher",
    "matcher",
    "_INCLUSION_WEIGHT",
    "_EXCLUSION_WEIGHT",
    "SUPPORTED_OPERATORS",
]

SUPPORTED_OPERATORS: List[str] = [
    "EQ",
    "GT",
    "LT",
    "GTE",
    "LTE",
    "IN",
    "NOT_IN",
    "CONTAINS",
    "BETWEEN",
    "EXISTS",
    "NOT_EXISTS",
]
