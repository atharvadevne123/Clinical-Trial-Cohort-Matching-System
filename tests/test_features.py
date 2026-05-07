"""Tests for feature engineering pipeline."""
import numpy as np
import pytest

from src.features import (
    ClinicalFeaturePipeline,
    build_feature_vector,
    compute_age,
    extract_condition_flags,
)


def test_extract_condition_flags_diabetes():
    conds = [{"code": "E11", "name": "diabetes"}]
    flags = extract_condition_flags(conds)
    assert flags["has_diabetes"] == 1
    assert flags["has_cancer"] == 0


def test_extract_condition_flags_empty():
    flags = extract_condition_flags([])
    for v in flags.values():
        assert v == 0


@pytest.mark.parametrize("condition,flag", [
    ([{"code": "I10"}], "has_hypertension"),
    ([{"code": "C50"}], "has_cancer"),
    (["atrial fibrillation"], "has_afib"),
    ([{"code": "I50"}], "has_heart_disease"),
])
def test_extract_parametrized_flags(condition, flag):
    flags = extract_condition_flags(condition)
    assert flags[flag] == 1


def test_compute_age_from_string():
    age = compute_age("1985-01-01")
    assert 35 < age < 45


def test_compute_age_none_returns_default():
    age = compute_age(None)
    assert age == 50.0


def test_compute_age_invalid_string():
    age = compute_age("not-a-date")
    assert age == 50.0


def test_build_feature_vector_shape():
    patient = {
        "date_of_birth": "1975-06-15",
        "gender": "male",
        "conditions": [{"code": "I10"}],
        "medications": [{"code": "C09AA01"}],
    }
    v = build_feature_vector(patient)
    assert v.shape == (14,)
    assert v.dtype == np.float32


def test_build_feature_vector_defaults():
    v = build_feature_vector({})
    assert v[10] == 25.0  # bmi default
    assert v[12] == 50.0  # distance default


def test_pipeline_fit_transform():
    patients = [
        {"date_of_birth": "1970-01-01", "gender": "male",
         "conditions": [{"code": "I10"}], "medications": []},
        {"date_of_birth": "1980-06-15", "gender": "female",
         "conditions": [], "medications": []},
    ]
    pipeline = ClinicalFeaturePipeline()
    X = pipeline.fit_transform(patients)
    assert X.shape == (2, 14)


def test_pipeline_transform_without_fit():
    pipeline = ClinicalFeaturePipeline()
    patients = [{"gender": "male", "conditions": [], "medications": []}]
    X = pipeline.transform(patients)
    assert X.shape == (1, 14)
