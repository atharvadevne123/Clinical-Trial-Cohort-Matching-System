"""Extended tests for features module — compute_age type annotations and edge cases."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.features import CONDITION_FLAGS, ClinicalFeaturePipeline, build_feature_vector, compute_age


class TestComputeAgeExtended:
    def test_none_returns_default_age(self):
        assert compute_age(None) == 50.0

    def test_datetime_object_returns_positive_age(self):
        dob = datetime(1990, 1, 1)
        age = compute_age(dob)
        assert age > 0

    def test_iso_string_parsed_correctly(self):
        age = compute_age("1990-01-01T00:00:00")
        assert age > 30

    def test_iso_string_with_z_suffix(self):
        age = compute_age("1985-06-15T00:00:00Z")
        assert age > 0

    def test_invalid_string_returns_default(self):
        age = compute_age("not-a-date")
        assert age == 50.0

    def test_timezone_aware_datetime(self):
        dob = datetime(1980, 3, 15, tzinfo=timezone.utc)
        age = compute_age(dob)
        assert age > 40


class TestBuildFeatureVector:
    BASE_PATIENT = {
        "date_of_birth": "1970-01-01T00:00:00",
        "gender": "male",
        "conditions": [{"code": "I10", "name": "Hypertension"}],
        "medications": [],
    }

    def test_returns_14_dimensional_array(self):
        vec = build_feature_vector(self.BASE_PATIENT)
        assert vec.shape == (14,)

    def test_male_gender_encodes_as_1(self):
        vec = build_feature_vector(self.BASE_PATIENT)
        assert vec[1] == 1.0

    def test_female_gender_encodes_as_0(self):
        p = {**self.BASE_PATIENT, "gender": "female"}
        vec = build_feature_vector(p)
        assert vec[1] == 0.0

    def test_num_conditions_matches_list_length(self):
        p = {**self.BASE_PATIENT, "conditions": [{"code": "I10"}, {"code": "E11"}]}
        vec = build_feature_vector(p)
        assert vec[2] == 2.0

    def test_bmi_defaults_to_25(self):
        vec = build_feature_vector(self.BASE_PATIENT)
        assert vec[10] == 25.0


class TestConditionFlags:
    def test_all_expected_flags_present(self):
        assert set(CONDITION_FLAGS.keys()) == {
            "has_diabetes",
            "has_hypertension",
            "has_heart_disease",
            "has_cancer",
            "has_afib",
        }

    @pytest.mark.parametrize(
        "condition,flag",
        [
            ("hypertension", "has_hypertension"),
            ("cancer", "has_cancer"),
            ("diabetes", "has_diabetes"),
        ],
    )
    def test_flag_detected_by_keyword(self, condition, flag):
        from src.features import extract_condition_flags

        conditions = [{"code": "X", "name": condition}]
        flags = extract_condition_flags(conditions)
        assert flags[flag] == 1


class TestClinicalFeaturePipeline:
    def test_fit_transform_returns_correct_shape(self):
        patients = [
            {"date_of_birth": "1970-01-01", "gender": "male", "conditions": [], "medications": []},
            {
                "date_of_birth": "1990-05-10",
                "gender": "female",
                "conditions": [],
                "medications": [],
            },
        ]
        pipeline = ClinicalFeaturePipeline()
        X = pipeline.fit_transform(patients)
        assert X.shape == (2, 14)

    def test_transform_after_fit_returns_same_shape(self):
        patients = [
            {"date_of_birth": "1975-01-01", "gender": "male", "conditions": [], "medications": []}
        ]
        pipeline = ClinicalFeaturePipeline()
        pipeline.fit_transform(patients)
        X = pipeline.transform(patients)
        assert X.shape == (1, 14)
