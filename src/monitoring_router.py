"""Monitoring API router for drift detection and metrics endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter
from src.monitoring import monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/drift", summary="Check prediction drift using KS test")
def check_drift() -> Dict[str, Any]:
    """Run the KS drift test between the reference and rolling window distributions.

    Returns:
        Drift result dict with drift_detected, ks_statistic, p_value, and sample_size.
    """
    return monitor.check_drift()


@router.get("/summary", summary="Prediction distribution summary statistics")
def distribution_summary() -> Dict[str, Any]:
    """Return descriptive statistics for the current rolling prediction window.

    Returns:
        Dict with count, mean, std, min, max of recent enrollment probabilities.
    """
    return monitor.summary()


@router.post("/reset", summary="Reset the prediction monitor window")
def reset_monitor() -> Dict[str, str]:
    """Clear the rolling prediction window and reset drift tracking.

    Returns:
        Confirmation message.
    """
    monitor.reset()
    logger.info("Prediction monitor reset via API.")
    return {"status": "reset", "message": "Prediction monitor window cleared."}
