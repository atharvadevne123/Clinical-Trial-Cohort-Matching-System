"""Drift monitoring for ML enrollment predictions.

Tracks prediction distributions over time and detects statistical drift
using the Kolmogorov-Smirnov test.
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DRIFT_THRESHOLD: float = 0.05


class PredictionMonitor:
    """Monitors ML enrollment prediction distributions for statistical drift.

    Maintains a rolling window of predictions and reference distribution,
    applying the KS test to detect significant distribution shifts.

    Attributes:
        window_size: Maximum number of recent predictions to retain.
        reference: Optional reference distribution for comparison.
        predictions: Rolling deque of recent prediction probabilities.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self.reference: Optional[np.ndarray] = None
        self.predictions: Deque[float] = deque(maxlen=window_size)

    def set_reference(self, probabilities: List[float]) -> None:
        """Set the reference distribution for drift comparison.

        Args:
            probabilities: List of enrollment probability floats (0.0–1.0)
                from a baseline period.
        """
        self.reference = np.array(probabilities, dtype=np.float64)
        logger.info("Reference distribution set with %d samples.", len(probabilities))

    def record(self, probability: float) -> None:
        """Record a single prediction probability into the rolling window.

        Args:
            probability: Enrollment probability from the ML model (0.0–1.0).
        """
        self.predictions.append(float(probability))

    def check_drift(self) -> Dict[str, object]:
        """Run the KS test between the rolling window and reference distribution.

        Returns:
            Dict with drift_detected (bool), ks_statistic, p_value, and
            sample_size. Returns drift_detected=False with zeros if insufficient data.
        """
        if self.reference is None or len(self.predictions) < 30:
            return {
                "drift_detected": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "sample_size": len(self.predictions),
                "message": "Insufficient data for drift detection.",
            }

        from scipy.stats import ks_2samp

        current = np.array(list(self.predictions))
        stat, p_value = ks_2samp(self.reference, current)
        drift = bool(p_value < _DRIFT_THRESHOLD)

        if drift:
            logger.warning(
                "Drift detected! KS statistic=%.4f, p-value=%.6f", stat, p_value
            )
        else:
            logger.debug("No drift detected. KS=%.4f, p=%.4f", stat, p_value)

        return {
            "drift_detected": drift,
            "ks_statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "sample_size": len(self.predictions),
            "threshold": _DRIFT_THRESHOLD,
        }

    def summary(self) -> Dict[str, object]:
        """Return descriptive statistics for the current rolling window.

        Returns:
            Dict with mean, std, min, max, and count of current predictions.
        """
        if not self.predictions:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        arr = np.array(list(self.predictions))
        return {
            "count": len(arr),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }

    def reset(self) -> None:
        """Clear the rolling window of recorded predictions."""
        self.predictions.clear()
        logger.info("Prediction monitor window reset.")


monitor = PredictionMonitor()
