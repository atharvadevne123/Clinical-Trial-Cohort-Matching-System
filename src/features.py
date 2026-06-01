"""Feature engineering pipeline for clinical trial cohort matching.

Provides transformers and helpers for constructing patient feature vectors
suitable for the XGBoost enrollment prediction model.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

PatientDict = Dict[str, Any]
ConditionList = List[Any]

__all__ = [
    "compute_age",
    "build_feature_vector",
    "extract_condition_flags",
    "ClinicalFeaturePipeline",
    "DEFAULT_AGE",
    "CONDITION_FLAGS",
    "_strip_timezone",
]


def _strip_timezone(dt: datetime) -> datetime:
    """Return a naive datetime by stripping timezone info if present."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


DEFAULT_AGE: float = 50.0
"""Fallback age in years used when date_of_birth is unknown or unparseable."""

CONDITION_FLAGS: Dict[str, List[str]] = {
    "has_diabetes": ["diabetes", "e11", "a10ba02"],
    "has_hypertension": ["hypertension", "i10", "c09aa01"],
    "has_heart_disease": ["heart disease", "coronary", "i50", "i25", "heart failure"],
    "has_cancer": ["cancer", "carcinoma", "c50", "c34", "c61", "c18", "c80"],
    "has_afib": ["atrial fibrillation", "afib", "i48", "b01aa03", "b01af02"],
}


def extract_condition_flags(conditions: List[Any]) -> Dict[str, int]:
    """Extract binary condition flag features from a patient conditions list.

    Args:
        conditions: List of condition dicts (with code/name keys) or code strings.

    Returns:
        Dict mapping flag name to 1 (present) or 0 (absent).
    """
    joined = " ".join(str(c).lower() for c in conditions)
    return {
        flag: int(any(kw in joined for kw in keywords))
        for flag, keywords in CONDITION_FLAGS.items()
    }


def compute_age(date_of_birth: Union[datetime, str, None]) -> float:
    """Compute patient age in years from date_of_birth.

    Args:
        date_of_birth: datetime, date, ISO string, or None.

    Returns:
        Age in decimal years, or 50.0 as a default if dob is unknown.
    """
    if date_of_birth is None:
        return DEFAULT_AGE
    if isinstance(date_of_birth, str):
        try:
            date_of_birth = datetime.fromisoformat(date_of_birth.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return DEFAULT_AGE
    dob_naive = _strip_timezone(date_of_birth)
    now = datetime.now()
    return (now - dob_naive).days / 365.25


def build_feature_vector(patient: Dict[str, Any]) -> np.ndarray:
    """Build the 14-dimensional feature vector for a patient dict.

    Args:
        patient: Patient record dict with date_of_birth, gender, conditions,
            medications, and optional fields.

    Returns:
        float32 numpy array of shape (14,) matching FEATURE_NAMES order.
    """
    conditions = patient.get("conditions") or []
    medications = patient.get("medications") or []
    flags = extract_condition_flags(conditions)

    return np.array(
        [
            compute_age(patient.get("date_of_birth")),
            1 if str(patient.get("gender", "")).upper() == "MALE" else 0,
            len(conditions),
            len(medications),
            flags["has_diabetes"],
            flags["has_hypertension"],
            flags["has_heart_disease"],
            flags["has_cancer"],
            flags["has_afib"],
            int(patient.get("smoker", False)),
            float(patient.get("bmi", 25.0)),
            int(patient.get("prior_trial_participation", False)),
            float(patient.get("distance_to_site_km", 50.0)),
            int(patient.get("num_exclusion_flags", 0)),
        ],
        dtype=np.float32,
    )


class ClinicalFeaturePipeline:
    """Sklearn-compatible transformer pipeline for patient feature extraction.

    Wraps build_feature_vector for batch processing with optional scaling.

    Attributes:
        scaler: Optional fitted StandardScaler instance.
    """

    def __init__(self) -> None:
        self.scaler: Optional[Any] = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the scaler has been fitted."""
        return self.scaler is not None

    @property
    def feature_names(self) -> List[str]:
        """Return the ordered list of feature names for the 14-dim vector."""
        return [
            "age", "gender_male", "num_conditions", "num_medications",
            "has_diabetes", "has_hypertension", "has_heart_disease", "has_cancer",
            "has_afib", "smoker", "bmi", "prior_trial_participation",
            "distance_to_site_km", "num_exclusion_flags",
        ]

    def reset(self) -> None:
        """Clear the fitted scaler so the pipeline can be re-fitted."""
        self.scaler = None

    def fit_transform(self, patients: List[Dict[str, Any]]) -> np.ndarray:
        """Build feature matrix and fit a StandardScaler.

        Args:
            patients: List of patient dicts.

        Returns:
            Scaled feature matrix of shape (n_patients, 14).
        """
        from sklearn.preprocessing import StandardScaler

        X = np.vstack([build_feature_vector(p) for p in patients])
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def transform(self, patients: List[Dict[str, Any]]) -> np.ndarray:
        """Transform patients using a pre-fitted scaler.

        Args:
            patients: List of patient dicts.

        Returns:
            Scaled feature matrix, or raw matrix if scaler is not fitted.
        """
        X = np.vstack([build_feature_vector(p) for p in patients])
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X
