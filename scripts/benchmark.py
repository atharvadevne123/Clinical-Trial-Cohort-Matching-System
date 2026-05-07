"""Performance benchmark utilities for the matching and prediction pipeline.

Provides timing helpers and batch throughput measurements for profiling
the eligibility matcher and ML predictor under realistic loads.
"""

import logging
import time
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def time_function(fn: Any, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Time a single function call.

    Args:
        fn: Callable to time.
        *args: Positional arguments for fn.
        **kwargs: Keyword arguments for fn.

    Returns:
        Tuple of (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_eligibility_matcher(n_patients: int = 100, n_trials: int = 10) -> Dict[str, Any]:
    """Benchmark the eligibility matcher across n_patients × n_trials pairs.

    Args:
        n_patients: Number of synthetic patient records to generate.
        n_trials: Number of synthetic trial records to generate.

    Returns:
        Dict with total_pairs, total_seconds, pairs_per_second, mean_ms.
    """
    from src.eligibility import EligibilityMatcher
    matcher = EligibilityMatcher()

    patients = [
        {
            "id": f"P{i:04d}",
            "date_of_birth": "1975-01-01T00:00:00",
            "gender": "male" if i % 2 == 0 else "female",
            "conditions": [{"code": "I10"}] if i % 3 == 0 else [],
            "medications": [{"code": "C09AA01"}] if i % 4 == 0 else [],
        }
        for i in range(n_patients)
    ]

    trials = [
        {
            "id": f"T{j:04d}",
            "inclusion_criteria": [{"field": "condition:I10", "operator": "EXISTS", "value": None}],
            "exclusion_criteria": [],
        }
        for j in range(n_trials)
    ]

    start = time.perf_counter()
    count = 0
    for patient in patients:
        for trial in trials:
            matcher.check_match(patient, trial)
            count += 1
    elapsed = time.perf_counter() - start

    throughput = count / elapsed if elapsed > 0 else float("inf")
    logger.info("Eligibility benchmark: %d pairs in %.3fs (%.0f pairs/s)", count, elapsed, throughput)

    return {
        "total_pairs": count,
        "total_seconds": round(elapsed, 4),
        "pairs_per_second": round(throughput, 1),
        "mean_ms": round(elapsed / count * 1000, 4) if count > 0 else 0.0,
    }


def benchmark_ml_predictor(n_patients: int = 100) -> Dict[str, Any]:
    """Benchmark the ML predictor for n_patients predictions.

    Args:
        n_patients: Number of patient records to predict.

    Returns:
        Dict with total_predictions, total_seconds, predictions_per_second, mean_ms.
    """
    from src.ml_prediction import predictor

    patients = [
        {
            "id": f"P{i:04d}",
            "date_of_birth": "1975-01-01T00:00:00",
            "gender": "male",
            "conditions": [],
            "medications": [],
        }
        for i in range(n_patients)
    ]

    start = time.perf_counter()
    predictor.predict_batch(patients, "BENCH_TRIAL")
    elapsed = time.perf_counter() - start

    throughput = n_patients / elapsed if elapsed > 0 else float("inf")
    logger.info("ML benchmark: %d predictions in %.3fs (%.0f/s)", n_patients, elapsed, throughput)

    return {
        "total_predictions": n_patients,
        "total_seconds": round(elapsed, 4),
        "predictions_per_second": round(throughput, 1),
        "mean_ms": round(elapsed / n_patients * 1000, 4),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json
    print("=== Eligibility Benchmark ===")
    print(json.dumps(benchmark_eligibility_matcher(200, 25), indent=2))
    print("\n=== ML Predictor Benchmark ===")
    print(json.dumps(benchmark_ml_predictor(200), indent=2))
