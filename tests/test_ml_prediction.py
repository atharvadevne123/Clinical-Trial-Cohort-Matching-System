"""Tests for ML enrollment prediction module."""

from unittest.mock import patch

import numpy as np
import pytest

from src.ml_prediction import (
    FEATURE_NAMES,
    EnrollmentPredictor,
    PatientFeatures,
    PredictionResult,
    _generate_training_data,
)


@pytest.fixture
def predictor():
    with patch("src.ml_prediction.MODEL_PATH", "/tmp/test_model.joblib"):
        p = EnrollmentPredictor()
    return p


@pytest.fixture
def sample_features():
    return PatientFeatures(
        age=55.0,
        gender_male=1,
        num_conditions=2,
        num_medications=3,
        has_diabetes=0,
        has_hypertension=1,
        has_heart_disease=0,
        has_cancer=0,
        has_afib=0,
        smoker=0,
        bmi=24.5,
        prior_trial_participation=1,
        distance_to_site_km=20.0,
        num_exclusion_flags=0,
    )


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 14


def test_patient_features_to_array(sample_features):
    arr = sample_features.to_array()
    assert arr.shape == (14,)
    assert arr.dtype == np.float32


def test_prediction_result_has_required_fields(predictor, sample_features):
    result = predictor.predict(sample_features, "P001", "T001")
    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.enrollment_probability <= 1.0
    assert result.confidence in ("HIGH", "MEDIUM", "LOW")
    assert result.patient_id == "P001"
    assert result.trial_id == "T001"
    assert isinstance(result.key_factors, list)


def test_prediction_probability_in_range(predictor, sample_features):
    result = predictor.predict(sample_features, "P001", "T001")
    assert 0.0 <= result.enrollment_probability <= 1.0


def test_batch_predict_sorted_by_probability(predictor):
    patients = [
        {
            "id": f"P{i}",
            "date_of_birth": "1970-01-01",
            "gender": "male",
            "conditions": [],
            "medications": [],
            "num_exclusion_flags": i,
        }
        for i in range(5)
    ]
    results = predictor.predict_batch(patients, "T001")
    probs = [r.enrollment_probability for r in results]
    assert probs == sorted(probs, reverse=True)


def test_generate_training_data_shape():
    X, y = _generate_training_data(100)
    assert X.shape == (100, 14)
    assert y.shape == (100,)
    assert set(y).issubset({0, 1})


@pytest.mark.parametrize(
    "age,expected_mention",
    [
        (55, "optimal range"),
        (80, "75"),
    ],
)
def test_explain_factors_age(predictor, age, expected_mention):
    f = PatientFeatures(age=age)
    factors = predictor._explain(f)
    factor_texts = " ".join(str(x) for x in factors)
    assert expected_mention in factor_texts


def test_recommendation_high_probability(predictor, sample_features):
    with patch.object(predictor, "_rule_based", return_value=0.85):
        predictor.model = None
        rec = predictor._recommendation(0.85, sample_features)
    assert "Strong" in rec


def test_recommendation_low_probability(predictor, sample_features):
    rec = predictor._recommendation(0.1, sample_features)
    assert "Low" in rec


def test_dict_to_features_parses_dob():
    patient = {
        "id": "P001",
        "date_of_birth": "1970-06-15T00:00:00",
        "gender": "female",
        "conditions": [{"code": "E11", "name": "Diabetes"}],
        "medications": [],
    }
    f = EnrollmentPredictor._dict_to_features(patient)
    assert 50 < f.age < 60
    assert f.gender_male == 0
    assert f.has_diabetes == 1


def test_dict_to_features_defaults_missing_fields():
    f = EnrollmentPredictor._dict_to_features({})
    assert f.age == 50.0
    assert f.bmi == 25.0
    assert f.distance_to_site_km == 50.0


@pytest.mark.parametrize(
    "gender,expected_male",
    [("male", 1), ("female", 0), ("MALE", 1), ("other", 0), ("", 0)],
)
def test_dict_to_features_gender_mapping(gender, expected_male):
    patient = {"gender": gender, "conditions": [], "medications": []}
    f = EnrollmentPredictor._dict_to_features(patient)
    assert f.gender_male == expected_male


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_predict_probability_bounds(predictor, prob):
    with patch.object(predictor, "_rule_based", return_value=prob):
        predictor.model = None
        f = PatientFeatures(age=45)
        result = predictor.predict(f, "P1", "T1")
    assert 0.0 <= result.enrollment_probability <= 1.0


def test_predict_confidence_levels(predictor):
    # HIGH: prob >= 0.75 or <= 0.25; MEDIUM: >= 0.60 or <= 0.40; LOW: otherwise
    for prob, expected_conf in [(0.9, "HIGH"), (0.65, "MEDIUM"), (0.5, "LOW")]:
        with patch.object(predictor, "_rule_based", return_value=prob):
            predictor.model = None
            f = PatientFeatures(age=45)
            result = predictor.predict(f, "P1", "T1")
        assert result.confidence == expected_conf


def test_patient_features_defaults():
    f = PatientFeatures()
    assert f.age == 50.0
    assert f.gender_male == 0
    assert f.bmi == 25.0


def test_prediction_result_all_fields_present(predictor, sample_features):
    result = predictor.predict(sample_features, "TEST_P", "TEST_T")
    assert hasattr(result, "enrollment_probability")
    assert hasattr(result, "confidence")
    assert hasattr(result, "recommendation")
    assert hasattr(result, "factors")
