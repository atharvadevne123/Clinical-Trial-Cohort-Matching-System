"""ML Enrollment Prediction — XGBoost classifier with joblib persistence."""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "enrollment_model.joblib")

FEATURE_NAMES = [
    "age", "gender_male", "num_conditions", "num_medications",
    "has_diabetes", "has_hypertension", "has_heart_disease",
    "has_cancer", "has_afib", "smoker", "bmi",
    "prior_trial_participation", "distance_to_site_km", "num_exclusion_flags",
]


@dataclass
class PatientFeatures:
    age: float = 0.0
    gender_male: int = 0
    num_conditions: int = 0
    num_medications: int = 0
    has_diabetes: int = 0
    has_hypertension: int = 0
    has_heart_disease: int = 0
    has_cancer: int = 0
    has_afib: int = 0
    smoker: int = 0
    bmi: float = 25.0
    prior_trial_participation: int = 0
    distance_to_site_km: float = 50.0
    num_exclusion_flags: int = 0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.age, self.gender_male, self.num_conditions, self.num_medications,
            self.has_diabetes, self.has_hypertension, self.has_heart_disease,
            self.has_cancer, self.has_afib, self.smoker, self.bmi,
            self.prior_trial_participation, self.distance_to_site_km,
            self.num_exclusion_flags,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return FEATURE_NAMES


@dataclass
class PredictionResult:
    patient_id: str
    trial_id: str
    enrollment_probability: float
    predicted_enrolled: bool
    confidence: str
    key_factors: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""
    predicted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ------------------------------------------------------------------
# Synthetic training data
# ------------------------------------------------------------------

def _generate_training_data(n: int = 1000):
    rng = np.random.default_rng(42)
    ages = rng.uniform(18, 80, n)
    gm = rng.integers(0, 2, n)
    nc = rng.integers(0, 8, n)
    nm = rng.integers(0, 10, n)
    diab = rng.integers(0, 2, n)
    htn = rng.integers(0, 2, n)
    hd = rng.integers(0, 2, n)
    ca = rng.integers(0, 2, n)
    af = rng.integers(0, 2, n)
    sm = rng.integers(0, 2, n)
    bmi = rng.uniform(17, 45, n)
    pt = rng.integers(0, 2, n)
    dist = rng.uniform(0, 200, n)
    ef = rng.integers(0, 5, n)

    score = (
        0.30 * ((ages >= 30) & (ages <= 70)).astype(float)
        + 0.10 * (nc < 5).astype(float)
        + 0.10 * (nm < 7).astype(float)
        - 0.20 * ca
        - 0.15 * (ef > 1).astype(float)
        - 0.10 * (dist > 100).astype(float)
        + 0.10 * pt
        + rng.normal(0, 0.1, n)
    )
    y = (score > 0.2).astype(int)
    X = np.column_stack([ages, gm, nc, nm, diab, htn, hd, ca, af, sm, bmi, pt, dist, ef]).astype(np.float32)
    return X, y


# ------------------------------------------------------------------
# Predictor
# ------------------------------------------------------------------

class EnrollmentPredictor:
    """XGBoost enrollment predictor with joblib model persistence."""

    def __init__(self):
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info("Loaded enrollment model from disk.")
                return
            except Exception as e:
                logger.warning(f"Could not load model ({e}), retraining.")
        self._train()

    def _train(self):
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed – using rule-based fallback.")
            return

        X, y = _generate_training_data(1000)
        split = int(0.8 * len(X))
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(
            X[:split], y[:split],
            eval_set=[(X[split:], y[split:])],
            verbose=False,
        )
        joblib.dump(self.model, MODEL_PATH)
        logger.info("Trained and saved enrollment model.")

    # ------------------------------------------------------------------

    def predict(self, features: PatientFeatures, patient_id: str, trial_id: str) -> PredictionResult:
        x = features.to_array().reshape(1, -1)
        prob = (
            float(self.model.predict_proba(x)[0][1])
            if self.model else self._rule_based(features)
        )
        confidence = (
            "HIGH" if prob >= 0.75 or prob <= 0.25
            else "MEDIUM" if prob >= 0.60 or prob <= 0.40
            else "LOW"
        )
        return PredictionResult(
            patient_id=patient_id,
            trial_id=trial_id,
            enrollment_probability=round(prob, 4),
            predicted_enrolled=prob >= 0.5,
            confidence=confidence,
            key_factors=self._explain(features),
            recommendation=self._recommendation(prob, features),
        )

    def predict_batch(self, patients: List[Dict[str, Any]], trial_id: str) -> List[PredictionResult]:
        results = [
            self.predict(self._dict_to_features(p), str(p.get("id", "unknown")), trial_id)
            for p in patients
        ]
        return sorted(results, key=lambda r: r.enrollment_probability, reverse=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rule_based(self, f: PatientFeatures) -> float:
        s = 0.5
        if 30 <= f.age <= 70:
            s += 0.10
        if f.num_conditions > 5:
            s -= 0.10
        if f.has_cancer:
            s -= 0.20
        if f.num_exclusion_flags > 1:
            s -= 0.15 * f.num_exclusion_flags
        if f.distance_to_site_km > 100:
            s -= 0.10
        if f.prior_trial_participation:
            s += 0.10
        return float(np.clip(s, 0.0, 1.0))

    def _explain(self, f: PatientFeatures) -> List[Dict[str, Any]]:
        factors: List[Dict[str, Any]] = []
        if 30 <= f.age <= 70:
            factors.append({"factor": "Age in optimal range (30–70)", "impact": "positive"})
        elif f.age > 75:
            factors.append({"factor": "Age above 75", "impact": "negative"})
        if f.has_cancer:
            factors.append({"factor": "Cancer diagnosis", "impact": "negative"})
        if f.has_afib:
            factors.append({"factor": "AFib present", "impact": "positive"})
        if f.num_exclusion_flags > 0:
            factors.append({"factor": f"{f.num_exclusion_flags} exclusion flag(s)", "impact": "negative"})
        if f.prior_trial_participation:
            factors.append({"factor": "Prior trial participation", "impact": "positive"})
        if f.distance_to_site_km > 100:
            factors.append({"factor": f"Distance {f.distance_to_site_km:.0f} km from site", "impact": "negative"})
        if self.model and hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            for i in np.argsort(imp)[::-1][:3]:
                factors.append({
                    "factor": f"ML: {FEATURE_NAMES[i]}",
                    "impact": "model",
                    "importance": round(float(imp[i]), 4),
                })
        return factors[:6]

    def _recommendation(self, prob: float, f: PatientFeatures) -> str:
        if prob >= 0.75:
            return "Strong candidate – prioritise outreach."
        if prob >= 0.55:
            return "Likely eligible – schedule screening."
        if prob >= 0.40:
            return "Borderline – review with PI."
        if f.num_exclusion_flags > 2:
            return "Multiple exclusion flags – confirm with PI."
        return "Low probability – consider future trials."

    @staticmethod
    def _dict_to_features(p: Dict[str, Any]) -> PatientFeatures:
        dob = p.get("date_of_birth")
        if isinstance(dob, (date, datetime)):
            ref = dob.date() if isinstance(dob, datetime) else dob
            age = (datetime.now().date() - ref).days / 365.25
        elif isinstance(dob, str):
            try:
                parsed = datetime.fromisoformat(dob.replace("Z", "+00:00")).replace(tzinfo=None)
                age = (datetime.now() - parsed).days / 365.25
            except (ValueError, TypeError):
                age = float(p.get("age", 50))
        else:
            age = float(p.get("age", 50))

        conds = p.get("conditions") or []
        meds = p.get("medications") or []

        def _has(lst: list, *keywords: str) -> int:
            joined = " ".join(str(item).lower() for item in lst)
            return int(any(kw in joined for kw in keywords))

        return PatientFeatures(
            age=age,
            gender_male=1 if str(p.get("gender", "")).upper() == "MALE" else 0,
            num_conditions=len(conds),
            num_medications=len(meds),
            has_diabetes=_has(conds, "diabetes", "e11"),
            has_hypertension=_has(conds, "hypertension", "i10"),
            has_heart_disease=_has(conds, "heart disease", "coronary", "i50", "i25"),
            has_cancer=_has(conds, "cancer", "carcinoma", "c50", "c34", "c61", "c18"),
            has_afib=_has(conds, "atrial fibrillation", "afib", "i48"),
            smoker=int(p.get("smoker", False)),
            bmi=float(p.get("bmi", 25.0)),
            prior_trial_participation=int(p.get("prior_trial_participation", False)),
            distance_to_site_km=float(p.get("distance_to_site_km", 50.0)),
            num_exclusion_flags=int(p.get("num_exclusion_flags", 0)),
        )


# Module-level singleton — loaded/trained once at import time
predictor = EnrollmentPredictor()


# ------------------------------------------------------------------
# FastAPI router factory
# ------------------------------------------------------------------

def create_ml_router():
    from fastapi import APIRouter
    from pydantic import BaseModel

    router = APIRouter(prefix="/ml", tags=["ML Prediction"])

    class SinglePredictRequest(BaseModel):
        patient_id: str
        trial_id: str
        features: Dict[str, Any]

    class BatchPredictRequest(BaseModel):
        patients: List[Dict[str, Any]]
        trial_id: str

    @router.post("/predict", summary="Predict enrollment probability for one patient")
    def predict_single(req: SinglePredictRequest):
        features = EnrollmentPredictor._dict_to_features(req.features)
        return predictor.predict(features, req.patient_id, req.trial_id).__dict__

    @router.post("/predict/batch", summary="Predict enrollment probability for a list of patients")
    def predict_batch(req: BatchPredictRequest):
        return [r.__dict__ for r in predictor.predict_batch(req.patients, req.trial_id)]

    @router.get("/model/info", summary="Model metadata and feature importances")
    def model_info():
        info: Dict[str, Any] = {
            "features": FEATURE_NAMES,
            "model_type": "XGBoost" if predictor.model else "RuleBased",
        }
        if predictor.model and hasattr(predictor.model, "feature_importances_"):
            info["feature_importances"] = {
                name: round(float(imp), 4)
                for name, imp in zip(FEATURE_NAMES, predictor.model.feature_importances_)
            }
        return info

    return router
