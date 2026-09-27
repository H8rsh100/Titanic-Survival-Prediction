"""Tests for the per-passenger explanation helpers."""

import numpy as np
import pandas as pd
import pytest

from explain import feature_drivers, logistic_member, nearest_passengers
from features import build_passenger_record


@pytest.fixture(scope="module")
def context(modeling_data):
    X_train, y_train, _x_test, _tids, _teids, train_eng, _te_eng = modeling_data
    return X_train, y_train, train_eng


@pytest.fixture(scope="module")
def model_and_names(repo_root):
    import os

    import joblib

    path = os.path.join(repo_root, "models", "best_model.pkl")
    if not os.path.exists(path):
        pytest.skip("Trained model artifact not present.")
    data = joblib.load(path)
    return data["model"], data["feature_names"]


class TestLogisticMember:
    def test_finds_the_logistic_estimator_in_the_ensemble(self, model_and_names):
        model, _ = model_and_names
        lr = logistic_member(model)
        assert lr is not None
        assert lr.__class__.__name__ == "LogisticRegression"

    def test_returns_none_for_a_tree_only_model(self):
        from sklearn.ensemble import RandomForestClassifier

        assert logistic_member(RandomForestClassifier(n_estimators=5)) is None


class TestFeatureDrivers:
    def test_returns_the_expected_columns(self, model_and_names, context):
        model, names = model_and_names
        X_train, _, _ = context
        row, _ = build_passenger_record(names, pclass=1, sex="female", title="Mrs", fare=100.0)
        drivers = feature_drivers(model, X_train, row)
        assert list(drivers.columns) == ["Feature", "Value", "Contribution"]

    def test_row_count_is_capped(self, model_and_names, context):
        model, names = model_and_names
        X_train, _, _ = context
        row, _ = build_passenger_record(names)
        assert len(feature_drivers(model, X_train, row, top_n=5)) == 5

    def test_sorted_by_absolute_contribution(self, model_and_names, context):
        model, names = model_and_names
        X_train, _, _ = context
        row, _ = build_passenger_record(names, pclass=1, sex="female", title="Mrs", fare=100.0)
        drivers = feature_drivers(model, X_train, row)
        magnitudes = drivers["Contribution"].abs().tolist()
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_values_are_finite(self, model_and_names, context):
        model, names = model_and_names
        X_train, _, _ = context
        row, _ = build_passenger_record(names)
        drivers = feature_drivers(model, X_train, row)
        assert np.isfinite(drivers["Contribution"]).all()

    def test_first_class_woman_is_pushed_toward_survival(self, model_and_names, context):
        model, names = model_and_names
        X_train, _, _ = context
        row, _ = build_passenger_record(names, pclass=1, sex="female", title="Mrs", fare=100.0)
        drivers = feature_drivers(model, X_train, row).set_index("Feature")
        assert drivers.loc["Sex", "Contribution"] > 0
        assert drivers.loc["Pclass", "Contribution"] > 0

    def test_third_class_man_is_pushed_toward_perishing(self, model_and_names, context):
        model, names = model_and_names
        X_train, _, _ = context
        row, _ = build_passenger_record(names, pclass=3, sex="male", title="Mr", fare=7.25)
        drivers = feature_drivers(model, X_train, row).set_index("Feature")
        assert drivers.loc["Sex", "Contribution"] < 0
        assert drivers.loc["Pclass", "Contribution"] < 0

    def test_one_hot_columns_are_collapsed_to_readable_names(self, model_and_names, context):
        model, names = model_and_names
        X_train, _, _ = context
        row, _ = build_passenger_record(names)
        features = feature_drivers(model, X_train, row)["Feature"].tolist()
        # Real engineered features such as Fare_Log keep their name; only the
        # indicator columns (Title_Mr, Deck_B, ...) must be collapsed away.
        allowed = {"Title", "Pclass", "Sex", "Embarked", "Deck", "AgeGroup",
                   "IsAlone", "LargeFamily", "Fare_Log", "Age", "Fare",
                   "SibSp", "Parch", "FamilySize", "PassengerId"}
        unexpected = [f for f in features if f not in allowed]
        assert not unexpected, f"raw dummy names leaked: {unexpected}"
        assert "Pclass" in features

    def test_empty_frame_without_logistic_member(self, context):
        from sklearn.ensemble import RandomForestClassifier

        X_train, _, _ = context
        row, _ = build_passenger_record(list(X_train.columns))
        drivers = feature_drivers(RandomForestClassifier(n_estimators=5, random_state=0),
                                  X_train, row)
        assert drivers.empty


class TestNearestPassengers:
    def test_returns_the_requested_number_of_rows(self, model_and_names, context):
        model, names = model_and_names
        X_train, y_train, train_eng = context
        row, _ = build_passenger_record(names)
        out = nearest_passengers(X_train, y_train, train_eng, row, k=6)
        assert len(out) == 6

    def test_similarity_is_bounded_and_descending(self, model_and_names, context):
        model, names = model_and_names
        X_train, y_train, train_eng = context
        row, _ = build_passenger_record(names)
        out = nearest_passengers(X_train, y_train, train_eng, row, k=8)
        sims = out["Similarity"].tolist()
        assert all(0.0 <= s <= 1.0 for s in sims)
        assert sims == sorted(sims, reverse=True)

    def test_closest_match_is_the_passenger_itself_when_scoring_a_training_row(
        self, model_and_names, context
    ):
        model, names = model_and_names
        X_train, y_train, train_eng = context
        row = X_train.iloc[[0]]
        out = nearest_passengers(X_train, y_train, train_eng, row, k=1)
        assert int(out["PassengerId"].iloc[0]) == int(train_eng["PassengerId"].iloc[0])
        assert out["Similarity"].iloc[0] == pytest.approx(1.0)

    def test_result_carries_the_survival_label(self, model_and_names, context):
        model, names = model_and_names
        X_train, y_train, train_eng = context
        row, _ = build_passenger_record(names, pclass=1, sex="female", title="Mrs")
        out = nearest_passengers(X_train, y_train, train_eng, row, k=10)
        assert "Survived" in out.columns
        assert set(out["Survived"].unique()) <= {0, 1}

    def test_k_larger_than_dataset_is_clamped(self, model_and_names, context):
        model, names = model_and_names
        X_train, y_train, train_eng = context
        row, _ = build_passenger_record(names)
        out = nearest_passengers(X_train, y_train, train_eng, row, k=len(X_train) + 50)
        assert len(out) == len(X_train)
