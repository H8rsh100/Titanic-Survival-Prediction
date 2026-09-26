"""Tests for raw data loading and leakage-aware missing value imputation."""

import os

import numpy as np
import pandas as pd
import pytest

from data_prep import impute_missing, load_data


class TestLoadData:
    def test_returns_expected_shapes(self, repo_root):
        train, test = load_data(os.path.join(repo_root, "data", "raw"))
        assert train.shape == (891, 12)
        assert test.shape == (418, 11)

    def test_train_has_target_and_test_does_not(self, repo_root):
        train, test = load_data(os.path.join(repo_root, "data", "raw"))
        assert "Survived" in train.columns
        assert "Survived" not in test.columns

    def test_loader_does_not_mutate_inputs(self, raw_data):
        train, test = raw_data
        train_before = train.copy()
        test_before = test.copy()
        load_data()
        pd.testing.assert_frame_equal(train, train_before)
        pd.testing.assert_frame_equal(test, test_before)


class TestImputation:
    def test_no_missing_values_remain_in_modeled_columns(self, imputed_data):
        train_clean, _ = imputed_data
        for column in ["Age", "Fare", "Embarked"]:
            assert not train_clean[column].isnull().any(), f"{column} still has nulls"

    def test_row_counts_are_preserved(self, raw_data, imputed_data):
        train, test = raw_data
        train_clean, test_clean = imputed_data
        assert len(train_clean) == len(train)
        assert len(test_clean) == len(test)

    def test_temp_title_helper_column_is_removed(self, imputed_data):
        train_clean, test_clean = imputed_data
        assert "TempTitle" not in train_clean.columns
        assert "TempTitle" not in test_clean.columns

    def test_observed_values_are_left_untouched(self, raw_data, imputed_data):
        train, _ = raw_data
        train_clean, _ = imputed_data
        observed = train.dropna(subset=["Age", "Fare", "Embarked"])
        merged = train_clean.merge(
            observed[["PassengerId", "Age", "Fare", "Embarked"]],
            on="PassengerId",
            suffixes=("_new", "_old"),
        )
        assert len(merged) == len(observed)
        np.testing.assert_allclose(merged["Age_new"], merged["Age_old"])
        np.testing.assert_allclose(merged["Fare_new"], merged["Fare_old"])
        assert (merged["Embarked_new"] == merged["Embarked_old"]).all()

    def test_imputed_ages_stay_within_plausible_bounds(self, imputed_data):
        train_clean, _ = imputed_data
        assert train_clean["Age"].between(0, 80).all()

    def test_embarked_imputation_uses_train_mode(self, raw_data, imputed_data):
        train, _ = raw_data
        _, test_clean = imputed_data
        train_mode = train["Embarked"].mode()[0]
        assert test_clean["Embarked"].mode()[0] == train_mode

    def test_fare_imputation_follows_pclass_median(self, raw_data, imputed_data):
        train, test = raw_data
        _, test_clean = imputed_data
        medians = train.groupby("Pclass")["Fare"].median()
        for _, row in test_clean.iterrows():
            if pd.isnull(test.loc[row.name, "Fare"]):
                assert row["Fare"] == pytest.approx(medians[row["Pclass"]])

    def test_inputs_are_not_mutated_in_place(self, raw_data):
        train, test = raw_data
        train_before, test_before = train.copy(), test.copy()
        impute_missing(train, test)
        pd.testing.assert_frame_equal(train, train_before)
        pd.testing.assert_frame_equal(test, test_before)

    def test_output_is_a_copy_not_a_view(self, raw_data):
        train, _ = raw_data
        train_clean, _ = impute_missing(train, raw_data[1])
        train_clean.loc[0, "Age"] = -1.0
        assert not np.isclose(train.loc[0, "Age"], -1.0)
