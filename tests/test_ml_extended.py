"""Extended ML prediction tests with parametrize."""

from unittest.mock import patch

import pytest

from src.ml_prediction import (
    EnrollmentPredictor,
    PatientFeatures,
    _generate_training_data,
)


@pytest.fixture
def predictor():
    with patch("src.ml_prediction.MODEL_PATH", "/tmp/test_model_ext.joblib"):
        p = EnrollmentPredictor()
    return p


@pytest.mark.parametrize(
    "age,has_cancer,expected_sign",
    [
        (55, 0, "positive_age"),
        (80, 0, "negative_age"),
        (55, 1, "cancer"),
    ],
)
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


@pytest.mark.parametrize(
    "prob,expected_rec_keyword",
    [
        (0.80, "Strong"),
        (0.60, "Likely"),
        (0.45, "Borderline"),
        (0.10, "Low"),
    ],
)
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


def test_dict_to_features_raises_on_non_dict():
    """_dict_to_features must raise TypeError for non-dict input."""
    with pytest.raises(TypeError, match="Expected dict"):
        EnrollmentPredictor._dict_to_features("not a dict")


def test_dict_to_features_raises_on_list():
    with pytest.raises(TypeError):
        EnrollmentPredictor._dict_to_features([1, 2, 3])


def test_predict_with_no_model_uses_rule_based(predictor):
    """When model is None, predict() should use _rule_based fallback."""
    predictor.model = None
    f = PatientFeatures(age=55)
    result = predictor.predict(f, "P001", "T001")
    assert 0.0 <= result.enrollment_probability <= 1.0
    assert result.confidence in ("HIGH", "MEDIUM", "LOW")


def test_predict_batch_result_sorted_descending(predictor):
    predictor.model = None
    patients = [
        {
            "id": f"P{i}",
            "date_of_birth": "1975-01-01",
            "gender": "male",
            "conditions": [],
            "medications": [],
            "bmi": 20 + i,
        }
        for i in range(5)
    ]
    results = predictor.predict_batch(patients, "T_BATCH")
    probs = [r.enrollment_probability for r in results]
    assert probs == sorted(probs, reverse=True)


@pytest.mark.parametrize(
    "age,expected_range",
    [
        (25, (0.0, 1.0)),
        (55, (0.0, 1.0)),
        (85, (0.0, 1.0)),
    ],
)
def test_rule_based_returns_valid_probability(predictor, age, expected_range):
    predictor.model = None
    f = PatientFeatures(age=age)
    prob = predictor._rule_based(f)
    assert expected_range[0] <= prob <= expected_range[1]


def test_predict_batch_empty_list(predictor):
    predictor.model = None
    results = predictor.predict_batch([], "T_EMPTY")
    assert results == []


def test_predict_batch_single_patient(predictor):
    predictor.model = None
    patients = [
        {
            "id": "P_SINGLE",
            "date_of_birth": "1975-01-01",
            "gender": "male",
            "conditions": [],
            "medications": [],
        }
    ]
    results = predictor.predict_batch(patients, "T_SINGLE")
    assert len(results) == 1
    assert results[0].patient_id == "P_SINGLE"


@pytest.mark.parametrize("num_patients", [1, 3, 5, 10])
def test_predict_batch_result_count(predictor, num_patients):
    predictor.model = None
    patients = [{"id": f"P{i}", "conditions": [], "medications": []} for i in range(num_patients)]
    results = predictor.predict_batch(patients, "T_COUNT")
    assert len(results) == num_patients


def test_dict_to_features_multiple_conditions(predictor):
    patient = {
        "conditions": [
            {"code": "E11", "name": "diabetes"},
            {"code": "I10", "name": "hypertension"},
            {"code": "C50", "name": "cancer"},
        ],
        "medications": [],
    }
    f = EnrollmentPredictor._dict_to_features(patient)
    assert f.has_diabetes == 1
    assert f.has_hypertension == 1
    assert f.has_cancer == 1
