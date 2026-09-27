"""Tests for the what-if sensitivity sweep."""

import pytest

from features import build_passenger_record
from sensitivity import SWEEPS, sensitivity_curve


@pytest.fixture(scope="module")
def model_and_names(repo_root):
    import os

    import joblib

    path = os.path.join(repo_root, "models", "best_model.pkl")
    if not os.path.exists(path):
        pytest.skip("Trained model artifact not present.")
    data = joblib.load(path)
    return data["model"], data["feature_names"]


@pytest.fixture
def base():
    return dict(
        pclass=3,
        sex="male",
        age=25.0,
        title="Mr",
        sibsp=0,
        parch=0,
        fare=15.0,
        embarked="S",
        cabin=None,
    )


class TestSweepCatalogue:
    def test_every_sweep_declares_an_attribute_and_values(self):
        for name, (attribute, values) in SWEEPS.items():
            assert isinstance(attribute, str) and attribute
            assert len(values) > 1, name

    def test_sweeps_are_non_decreasing(self):
        for name, (_, values) in SWEEPS.items():
            if all(isinstance(v, (int, float)) for v in values):
                assert list(values) == sorted(values), name


class TestSensitivityCurve:
    def test_returns_one_row_per_value(self, model_and_names, base):
        model, names = model_and_names
        curve, base_p = sensitivity_curve(model, names, base, "Fare")
        assert len(curve) == len(SWEEPS["Fare"][1])

    def test_columns_and_dtypes(self, model_and_names, base):
        model, names = model_and_names
        curve, _ = sensitivity_curve(model, names, base, "Age")
        assert list(curve.columns) == ["Value", "SurvivalProbability", "Delta"]
        assert curve["SurvivalProbability"].between(0, 1).all()

    def test_base_probability_matches_a_direct_predict(self, model_and_names, base):
        model, names = model_and_names
        _, base_p = sensitivity_curve(model, names, base, "Fare")
        row, _ = build_passenger_record(names, **base)
        assert base_p == pytest.approx(float(model.predict_proba(row)[0, 1]))

    def test_delta_is_relative_to_the_base_case(self, model_and_names, base):
        model, names = model_and_names
        curve, base_p = sensitivity_curve(model, names, base, "Pclass")
        assert base_p in curve["SurvivalProbability"].tolist()
        for _, r in curve.iterrows():
            assert r["Delta"] == pytest.approx(r["SurvivalProbability"] - base_p)

    def test_male_and_female_differ_under_a_sex_sweep(self, model_and_names, base):
        model, names = model_and_names
        curve, _ = sensitivity_curve(model, names, base, "Sex")
        assert curve["SurvivalProbability"].nunique() == 2
        female = curve.loc[curve["Value"] == "female", "SurvivalProbability"].iloc[0]
        male = curve.loc[curve["Value"] == "male", "SurvivalProbability"].iloc[0]
        assert female > male

    def test_first_class_beats_third_class_under_a_class_sweep(self, model_and_names, base):
        model, names = model_and_names
        curve, _ = sensitivity_curve(model, names, base, "Pclass")
        lookup = dict(zip(curve["Value"], curve["SurvivalProbability"]))
        assert lookup[1] > lookup[3]

    def test_higher_fare_raises_the_probability_overall(self, model_and_names, base):
        # Not strictly monotonic: the tree members step across thresholds, so
        # assert the overall direction rather than a smooth curve.
        model, names = model_and_names
        curve, _ = sensitivity_curve(model, names, base, "Fare")
        assert curve["SurvivalProbability"].iloc[-1] > curve["SurvivalProbability"].iloc[0]
        assert curve["SurvivalProbability"].max() > 0.4

    def test_higher_age_trends_downward(self, model_and_names, base):
        model, names = model_and_names
        curve, _ = sensitivity_curve(model, names, base, "Age")
        assert curve["SurvivalProbability"].iloc[-1] < curve["SurvivalProbability"].iloc[0]

    def test_base_kwargs_are_not_mutated(self, model_and_names, base):
        model, names = model_and_names
        snapshot = dict(base)
        sensitivity_curve(model, names, base, "Fare")
        assert base == snapshot

    def test_unknown_sweep_raises(self, model_and_names, base):
        model, names = model_and_names
        with pytest.raises(ValueError, match="Unknown sweep"):
            sensitivity_curve(model, names, base, "TicketNumber")

    def test_bulk_siblings_lower_the_probability(self, model_and_names, base):
        model, names = model_and_names
        curve, _ = sensitivity_curve(model, names, base, "SibSp")
        lookup = dict(zip(curve["Value"], curve["SurvivalProbability"]))
        assert lookup[8] < lookup[0]
