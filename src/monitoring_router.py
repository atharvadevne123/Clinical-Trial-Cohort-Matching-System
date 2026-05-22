"""Monitoring API router for drift detection and metrics endpoints."""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader

from src.monitoring import monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: Optional[str] = Security(_api_key_header)) -> Optional[str]:
    """Validate the X-API-Key header for monitoring endpoints."""
    api_key = os.environ.get("API_KEY", "")
    if api_key and key != api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")
    return key


@router.get("/drift", summary="Check prediction drift using KS test")
def check_drift(_key: Optional[str] = Security(_require_api_key)) -> Dict[str, Any]:
    """Run the KS drift test between the reference and rolling window distributions.

    Returns:
        Drift result dict with drift_detected, ks_statistic, p_value, and sample_size.
    """
    return monitor.check_drift()


@router.get("/summary", summary="Prediction distribution summary statistics")
def distribution_summary(_key: Optional[str] = Security(_require_api_key)) -> Dict[str, Any]:
    """Return descriptive statistics for the current rolling prediction window.

    Returns:
        Dict with count, mean, std, min, max of recent enrollment probabilities.
    """
    return monitor.summary()


@router.post("/reset", summary="Reset the prediction monitor window")
def reset_monitor(_key: Optional[str] = Security(_require_api_key)) -> Dict[str, str]:
    """Clear the rolling prediction window and reset drift tracking.

    Returns:
        Confirmation message.
    """
    monitor.reset()
    logger.info("Prediction monitor reset via API.")
    return {"status": "reset", "message": "Prediction monitor window cleared."}


@router.post("/set-reference", summary="Set the reference distribution for drift detection")
def set_reference(
    payload: Dict[str, Any],
    _key: Optional[str] = Security(_require_api_key),
) -> Dict[str, Any]:
    """Set the reference distribution from a list of enrollment probabilities.

    Args:
        payload: Dict with ``probabilities`` key containing a list of floats.

    Returns:
        Confirmation with count of reference samples set.

    Raises:
        HTTPException: 400 if ``probabilities`` key is missing or not a list.
    """
    probs = payload.get("probabilities")
    if not isinstance(probs, list):
        raise HTTPException(status_code=400, detail="'probabilities' must be a list of floats")
    monitor.set_reference([float(p) for p in probs])
    logger.info("Reference distribution set with %d samples via API.", len(probs))
    return {"status": "ok", "reference_samples": len(probs)}
