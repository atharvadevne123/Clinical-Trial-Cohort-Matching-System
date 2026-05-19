"""Automated model retraining pipeline.

Provides a pipeline that can be triggered manually or scheduled to retrain
the XGBoost enrollment model using recent prediction data from the database.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "src", "enrollment_model.joblib"
)


class RetrainingPipeline:
    """Orchestrates model retraining from stored prediction data.

    Attributes:
        model_path: Path to the serialised model file.
        min_samples: Minimum samples required to trigger retraining.
    """

    def __init__(self, model_path: str = MODEL_PATH, min_samples: int = 500) -> None:
        self.model_path = model_path
        self.min_samples = min_samples

    def run(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Retrain the XGBoost model on new data and persist the updated model.

        Args:
            X: Feature matrix of shape (n_samples, 14).
            y: Binary label array of shape (n_samples,).

        Returns:
            Dict with status, sample_count, train_auc, and timestamp.
        """
        if len(X) < self.min_samples:
            logger.warning(
                "Skipping retraining: %d samples < minimum %d.", len(X), self.min_samples
            )
            return {
                "status": "skipped",
                "reason": f"Insufficient samples ({len(X)} < {self.min_samples})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        try:
            import xgboost as xgb
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            logger.error("Required package missing: %s", exc)
            return {"status": "error", "reason": str(exc)}

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, val_proba))

        joblib.dump(model, self.model_path)
        logger.info("Model retrained: AUC=%.4f, samples=%d, path=%s", auc, len(X), self.model_path)

        return {
            "status": "success",
            "sample_count": len(X),
            "train_auc": round(auc, 4),
            "model_path": self.model_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def generate_synthetic_data(self, n: int = 1000) -> tuple:
        """Generate synthetic training data for pipeline testing.

        Args:
            n: Number of samples to generate.

        Returns:
            Tuple (X, y) of feature matrix and binary labels.
        """
        from src.ml_prediction import _generate_training_data
        return _generate_training_data(n)


pipeline = RetrainingPipeline()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = RetrainingPipeline()
    X, y = p.generate_synthetic_data(1000)
    result = p.run(X, y)
    logger.info("Retraining result: %s", json.dumps(result, indent=2))
