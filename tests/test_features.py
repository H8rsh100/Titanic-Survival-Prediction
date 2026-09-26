"""Tests for domain feature engineering and the modelling matrix encoder."""

import numpy as np
import pandas as pd
import pytest

from features import extract_features, prepare_modeling_data


def make_frame(**overrides):
    """Build a minimal passenger frame containing every column extract_features needs.

    Scalar overrides are broadcast across the row count implied by the longest
    supplied column, so a test only has to state the fields it cares about.
    """
    defaults = {
        "Name": "Mr. Placeholder",
        "SibSp": 0,
        "Parch": 0,
        "Cabin": None,
        "Age": 30.0,
        "Fare": 10.0,
    }
    n_rows = max((len(value) for value in overrides.values()), default=1)

    records = {}
    for key, value in {**defaults, **overrides}.items():
        if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
            assert len(value) == n_rows, f"{key} length does not match row count"
            records[key] = list(value)
        else:
            records[key] = [value] * n_rows
    return pd.DataFrame(records)


class TestTitleExtraction:
    def test_known_titles_are_recognised(self, imputed_data):
        train_clean, _ = imputed_data
        titles = set(extract_features(train_clean)["Title"].unique())
        assert {"Mr", "Miss", "Mrs", "Master", "Rare"} <= titles

    def test_noble_titles_are_collapsed_into_rare(self):
        frame = make_frame(
            Name=[
                "Rothschild, Mrs. Martin (Elizabeth A.)",
                "Rothschild, Master. Richard",
                "Harding, Miss. Anna",
                "Wallenberg, Mme. Alicia",
                "Skoog, Mlle. Britta",
            ]
        )
        titles = extract_features(frame)["Title"].tolist()
        assert titles == ["Mrs", "Master", "Miss", "Mrs", "Miss"]

    def test_unrecognised_titles_fall_back_to_rare(self):
        frame = make_frame(Name=["Ramirez, Zorro.", "Doe, Ms. Jane"])
        assert extract_features(frame)["Title"].tolist() == ["Rare", "Miss"]


class TestFamilyFeatures:
    def test_family_size_includes_the_passenger(self, imputed_data):
        train_clean, _ = imputed_data
        engineered = extract_features(train_clean)
        expected = engineered["SibSp"] + engineered["Parch"] + 1
        assert (engineered["FamilySize"] == expected).all()

    def test_family_buckets_are_mutually_exclusive(self, imputed_data):
        train_clean, _ = imputed_data
        engineered = extract_features(train_clean)
        buckets = engineered[["IsAlone", "SmallFamily", "LargeFamily"]].sum(axis=1)
        assert (buckets == 1).all()

    def test_bucket_boundaries(self):
        frame = make_frame(
            Name=["Mr. A", "Mr. B", "Mr. C", "Mr. D"],
            SibSp=[0, 1, 1, 3],
            Parch=[0, 0, 2, 2],
        )
        engineered = extract_features(frame)
        assert engineered["FamilySize"].tolist() == [1, 2, 4, 6]
        assert engineered["IsAlone"].tolist() == [1, 0, 0, 0]
        assert engineered["SmallFamily"].tolist() == [0, 1, 1, 0]
        assert engineered["LargeFamily"].tolist() == [0, 0, 0, 1]


class TestCabinAndFareFeatures:
    def test_missing_cabin_maps_to_unknown_deck(self, imputed_data):
        train_clean, _ = imputed_data
        engineered = extract_features(train_clean)
        no_cabin = engineered[engineered["Cabin"].isnull()]
        assert (no_cabin["Deck"] == "U").all()
        assert (no_cabin["HasCabin"] == 0).all()

    def test_cabin_letter_becomes_deck(self):
        frame = make_frame(Name=["Mr. A", "Mr. B", "Mr. C"], Cabin=["C85", "t", np.nan])
        engineered = extract_features(frame)
        assert engineered["Deck"].tolist() == ["C", "U", "U"]
        assert engineered["HasCabin"].tolist() == [1, 1, 0]

    def test_fare_log_is_finite_and_rank_preserving(self, imputed_data):
        train_clean, _ = imputed_data
        engineered = extract_features(train_clean)
        assert np.isfinite(engineered["Fare_Log"]).all()
        np.testing.assert_allclose(engineered["Fare_Log"], np.log1p(engineered["Fare"]))
        assert engineered["Fare_Log"].corr(engineered["Fare"], method="spearman") == 1.0


class TestAgeBinning:
    def test_boundaries_land_in_expected_groups(self):
        frame = make_frame(Name=["Mr. A"] * 5, Age=[1, 12, 18, 35, 60])
        groups = extract_features(frame)["AgeGroup"].astype(str).tolist()
        assert groups == ["Child", "Teen", "YoungAdult", "Adult", "Senior"]


class TestPrepareModelingData:
    def test_matrix_shapes_match_dataset_sizes(self, modeling_data):
        X_train, y_train, X_test, train_ids, test_ids, _, _ = modeling_data
        assert X_train.shape[0] == len(y_train) == len(train_ids) == 891
        assert X_test.shape[0] == len(test_ids) == 418

    def test_train_and_test_share_an_identical_feature_contract(self, modeling_data):
        X_train, _, X_test, *_ = modeling_data
        assert list(X_train.columns) == list(X_test.columns)

    def test_feature_matrix_is_fully_numeric_and_finite(self, modeling_data):
        X_train, _, X_test, *_ = modeling_data
        for matrix in (X_train, X_test):
            for column in matrix.columns:
                dtype = matrix[column].dtype
                assert np.issubdtype(dtype, np.number) or dtype == bool, (
                    f"{column} has non-numeric dtype {dtype}"
                )
            values = matrix.to_numpy(dtype=float)
            assert np.isfinite(values).all()

    def test_no_one_hot_column_is_all_zero(self, modeling_data):
        X_train, *_ = modeling_data
        dummy_columns = [c for c in X_train.columns if "_" in c and X_train[c].nunique() <= 2]
        for column in dummy_columns:
            assert X_train[column].sum() > 0, f"{column} is constant zero and adds no signal"

    def test_target_is_binary_and_ids_are_unique(self, modeling_data):
        _, y_train, _, train_ids, test_ids, *_ = modeling_data
        assert set(np.unique(y_train)) <= {0, 1}
        assert len(set(train_ids)) == len(train_ids)
        assert len(set(test_ids)) == len(test_ids)

    def test_raw_identifier_columns_are_dropped(self, modeling_data):
        X_train, *_ = modeling_data
        for dropped in ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]:
            assert dropped not in X_train.columns

    def test_engineered_frames_retain_survival_labels(self, modeling_data):
        *_, train_eng, _ = modeling_data
        assert "Survived" in train_eng.columns
        assert "Title" in train_eng.columns
