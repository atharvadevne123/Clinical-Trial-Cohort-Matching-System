"""Extended ML prediction tests with parametrize."""
import pytest
import numpy as np
from unittest.mock import patch
from src.ml_prediction import (
    EnrollmentPredictor, PatientFeatures, _generate_training_data, FEATURE_NAMES
)


@pytest.fixture
def predictor():
    with patch("src.ml_prediction.MODEL_PATH", "/tmp/test_model_ext.joblib"):
        p = EnrollmentPredictor()
    return p


@pytest.mark.parametrize("age,has_cancer,expected_sign", [
    (55, 0, "positive_age"),
    (80, 0, "negative_age"),
    (55, 1, "cancer"),
])
def test_explain_factors_coverage(predictor, age, has_cancer, expected_sign):
    f = PatientFeatures(age=age, has_cancer=has_cancer)
    factors = predictor._explain(f)
    factor_text = " ".join(str(x) for x in factors)
    if expected_sign == "positive_age":
        assert "optimal" in factor_text
    elif expected_sign == "negative_age":
        assert "75" in factor_text
    elif expected_sign == "cancer":
        assert "Cancer" in factor_text


@pytest.mark.parametrize("prob,expected_rec_keyword", [
    (0.80, "Strong"),
    (0.60, "Likely"),
    (0.45, "Borderline"),
    (0.10, "Low"),
])
def test_recommendation_strings(predictor, prob, expected_rec_keyword):
    f = PatientFeatures()
    rec = predictor._recommendation(prob, f)
    assert expected_rec_keyword in rec


def test_rule_based_falls_back_when_no_model(predictor):
    predictor.model = None
    f = PatientFeatures(age=55, prior_trial_participation=1)
    prob = predictor._rule_based(f)
    assert 0.0 <= prob <= 1.0


def test_rule_based_penalises_cancer(predictor):
    predictor.model = None
    f_with = PatientFeatures(age=55, has_cancer=1)
    f_without = PatientFeatures(age=55, has_cancer=0)
    assert predictor._rule_based(f_with) < predictor._rule_based(f_without)


def test_rule_based_penalises_exclusion_flags(predictor):
    predictor.model = None
    f_clean = PatientFeatures(num_exclusion_flags=0)
    f_flagged = PatientFeatures(num_exclusion_flags=3)
    assert predictor._rule_based(f_flagged) < predictor._rule_based(f_clean)


def test_training_data_labels_binary():
    X, y = _generate_training_data(200)
    assert set(y).issubset({0, 1})
    assert X.shape == (200, 14)


def test_dict_to_features_string_dob():
    p = {"date_of_birth": "1985-03-20", "gender": "male", "conditions": [], "medications": []}
    f = EnrollmentPredictor._dict_to_features(p)
    assert 35 < f.age < 45


def test_dict_to_features_condition_flags():
    p = {
        "conditions": [
            {"code": "E11", "name": "diabetes"},
            {"code": "C50", "name": "breast cancer"},
        ],
        "medications": [],
    }
    f = EnrollmentPredictor._dict_to_features(p)
    assert f.has_diabetes == 1
    assert f.has_cancer == 1
    assert f.has_hypertension == 0
