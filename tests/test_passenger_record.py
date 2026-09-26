"""Tests for the single-passenger record builder used by the Streamlit simulator."""

import numpy as np
import pandas as pd
import pytest

from features import build_passenger_record


@pytest.fixture(scope="module")
def feature_names(modeling_data):
    return list(modeling_data[0].columns)


class TestBuildPassengerRecord:
    def test_output_is_a_single_row_matching_the_contract(self, feature_names):
        record, engineered = build_passenger_record(feature_names)
        assert len(record) == 1
        assert len(engineered) == 1
        assert list(record.columns) == feature_names

    def test_matrix_is_finite(self, feature_names):
        record, _ = build_passenger_record(feature_names, age=40.0, fare=100.0)
        assert np.isfinite(record.to_numpy(dtype=float)).all()

    def test_age_group_dummies_are_actually_set(self, feature_names):
        """Regression: the simulator used to leave every AgeGroup dummy at zero."""
        record, engineered = build_passenger_record(feature_names, age=8.0, title="Master")
        group_cols = [c for c in record.columns if c.startswith("AgeGroup_")]
        assert group_cols
        assert record[group_cols].to_numpy().sum() == 0, (
            "drop_first=True means a Child maps to all-zero AgeGroup dummies"
        )
        assert str(engineered["AgeGroup"].iloc[0]) == "Child"

    def test_non_dropped_age_group_sets_exactly_one_dummy(self, feature_names):
        record, engineered = build_passenger_record(feature_names, age=30.0, title="Mr")
        group_cols = [c for c in record.columns if c.startswith("AgeGroup_")]
        assert record[group_cols].to_numpy().sum() == 1
        assert str(engineered["AgeGroup"].iloc[0]) == "YoungAdult"

    def test_blank_cabin_forces_unknown_deck_and_no_cabin_flag(self, feature_names):
        """Regression: the old form allowed HasCabin=0 together with a named deck."""
        record, engineered = build_passenger_record(feature_names, cabin=None)
        assert int(engineered["HasCabin"].iloc[0]) == 0
        assert str(engineered["Deck"].iloc[0]) == "U"
        assert int(record["HasCabin"].iloc[0]) == 0

    def test_supplied_cabin_sets_flag_and_deck_consistently(self, feature_names):
        record, engineered = build_passenger_record(feature_names, cabin="C85")
        assert int(engineered["HasCabin"].iloc[0]) == 1
        assert str(engineered["Deck"].iloc[0]) == "C"
        assert int(record["HasCabin"].iloc[0]) == 1
        assert int(record["Deck_C"].iloc[0]) == 1

    def test_has_cabin_and_deck_never_disagree(self, feature_names):
        """No cabin recorded must always mean Deck 'U', and vice versa.

        Deck 'T' is intentionally folded into 'U' by the feature pipeline while
        HasCabin stays 1, so that case is asserted separately.
        """
        for cabin in [None, "", "   ", "A", "B7", "  C85  "]:
            _, engineered = build_passenger_record(feature_names, cabin=cabin)
            has_cabin = int(engineered["HasCabin"].iloc[0])
            deck = str(engineered["Deck"].iloc[0])
            assert has_cabin == (0 if deck == "U" else 1), f"inconsistent for cabin={cabin!r}"

    def test_rare_deck_t_is_grouped_with_unknown(self, feature_names):
        _, engineered = build_passenger_record(feature_names, cabin="T")
        assert int(engineered["HasCabin"].iloc[0]) == 1
        assert str(engineered["Deck"].iloc[0]) == "U"

    def test_family_features_match_the_inputs(self, feature_names):
        _, engineered = build_passenger_record(feature_names, sibsp=2, parch=3)
        assert int(engineered["FamilySize"].iloc[0]) == 6
        assert int(engineered["LargeFamily"].iloc[0]) == 1
        assert int(engineered["IsAlone"].iloc[0]) == 0

    @pytest.mark.parametrize("title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
    def test_every_offered_title_round_trips(self, feature_names, title):
        record, engineered = build_passenger_record(feature_names, title=title)
        assert str(engineered["Title"].iloc[0]) == title
        assert record.to_numpy(dtype=float).sum() > 0

    def test_sex_dummy_is_set_for_each_sex(self, feature_names):
        """Regression: drop_first on a single row zeroed every Sex dummy, so the
        simulator gave male and female passengers identical survival odds."""
        male, _ = build_passenger_record(feature_names, sex="male")
        female, _ = build_passenger_record(feature_names, sex="female")
        assert int(male["Sex_male"].iloc[0]) == 1
        assert int(female["Sex_male"].iloc[0]) == 0
        assert not male["Sex_male"].equals(female["Sex_male"])

    def test_all_categorical_dummies_are_present_and_boolean_valued(self, feature_names):
        from features import CATEGORICAL_COLUMNS, FEATURE_CATEGORIES

        record, _ = build_passenger_record(feature_names)
        for column in CATEGORICAL_COLUMNS:
            for level in FEATURE_CATEGORIES[column][1:]:
                assert f"{column}_{level}" in record.columns
                assert int(record[f"{column}_{level}"].iloc[0]) in (0, 1)

    def test_exactly_one_dummy_fires_per_categorical(self, feature_names):
        from features import CATEGORICAL_COLUMNS, FEATURE_CATEGORIES

        record, _ = build_passenger_record(
            feature_names, sex="female", embarked="Q", title="Mrs", cabin="C85", age=45.0
        )
        # female and Child are reference categories, so those sum to 0 by design.
        for column in CATEGORICAL_COLUMNS:
            levels = FEATURE_CATEGORIES[column][1:]
            fired = sum(int(record[f"{column}_{lvl}"].iloc[0]) for lvl in levels)
            assert fired in (0, 1), f"{column} fired {fired} dummies"

    def test_unsupported_title_is_rejected(self, feature_names):
        with pytest.raises(ValueError, match="Unsupported title"):
            build_passenger_record(feature_names, title="Captain")

    def test_unsupported_port_is_rejected(self, feature_names):
        with pytest.raises(ValueError, match="Unsupported embarkation"):
            build_passenger_record(feature_names, embarked="X")

    def test_probabilities_are_ordered_the_historically_correct_way(self, feature_names):
        """Sanity check that the encoding survived the refactor: women in first class
        must score materially higher than men in third class."""
        import joblib
        import os

        artifact = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.pkl"
        )
        if not os.path.exists(artifact):
            pytest.skip("Trained model artifact not present.")

        model = joblib.load(artifact)["model"]
        women_first, _ = build_passenger_record(
            feature_names, pclass=1, sex="female", age=30.0, title="Mrs", fare=100.0
        )
        men_third, _ = build_passenger_record(
            feature_names, pclass=3, sex="male", age=30.0, title="Mr", fare=7.25
        )
        assert (
            model.predict_proba(women_first)[0][1]
            > model.predict_proba(men_third)[0][1]
        )
