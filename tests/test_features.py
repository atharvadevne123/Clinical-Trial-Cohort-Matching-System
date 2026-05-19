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


def test_compute_age_timezone_aware_dob():
    """compute_age should handle timezone-aware datetime objects."""
    from datetime import datetime, timezone
    dob_aware = datetime(1985, 3, 10, tzinfo=timezone.utc)
    age = compute_age(dob_aware)
    assert 35 < age < 45


def test_compute_age_from_datetime_object():
    from datetime import datetime
    dob = datetime(1980, 1, 1)
    age = compute_age(dob)
    assert 40 < age < 50


@pytest.mark.parametrize("dob,expected_range", [
    ("1950-01-01", (70, 80)),
    ("2000-01-01", (20, 30)),
    ("1990-06-15", (30, 40)),
])
def test_compute_age_parametrized(dob, expected_range):
    age = compute_age(dob)
    assert expected_range[0] < age < expected_range[1]


def test_build_feature_vector_female_gender():
    patient = {"gender": "female", "conditions": [], "medications": []}
    v = build_feature_vector(patient)
    assert v[1] == 0  # gender_male should be 0 for female


def test_build_feature_vector_gender_case_insensitive():
    patient_male = {"gender": "MALE", "conditions": [], "medications": []}
    v = build_feature_vector(patient_male)
    assert v[1] == 1


def test_strip_timezone_exported():
    """Verify _strip_timezone helper is accessible and works."""
    from datetime import datetime, timezone
    from src.features import _strip_timezone
    aware = datetime(2020, 1, 1, tzinfo=timezone.utc)
    naive = _strip_timezone(aware)
    assert naive.tzinfo is None
    already_naive = datetime(2020, 1, 1)
    assert _strip_timezone(already_naive).tzinfo is None


def test_features_all_exports():
    """Key symbols should appear in __all__."""
    from src.features import __all__ as exports
    assert "compute_age" in exports
    assert "build_feature_vector" in exports
    assert "DEFAULT_AGE" in exports


def test_default_age_constant():
    from src.features import DEFAULT_AGE
    assert DEFAULT_AGE == 50.0


@pytest.mark.parametrize("condition_list,expected_flag,expected_value", [
    ([{"code": "I10"}], "has_hypertension", 1),
    ([{"code": "I48"}], "has_afib", 1),
    ([], "has_diabetes", 0),
    ([{"name": "heart failure"}], "has_heart_disease", 1),
])
def test_extract_condition_flags_extended(condition_list, expected_flag, expected_value):
    flags = extract_condition_flags(condition_list)
    assert flags[expected_flag] == expected_value
