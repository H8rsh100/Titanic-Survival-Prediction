"""Tests for batch manifest normalisation, encoding and scoring."""

import numpy as np
import pandas as pd
import pytest

from manifest import (
    ManifestError, normalize_manifest, prepare_manifest, score_manifest,
    summarise_predictions, validate_manifest
)


def compact_manifest(**overrides):
    frame = pd.DataFrame({
        "Pclass": [1, 3],
        "Sex": ["female", "male"],
        "Age": [30.0, 45.0],
        "SibSp": [0, 1],
        "Parch": [0, 0],
        "Fare": [100.0, 7.25],
        "Embarked": ["C", "S"],
        "Title": ["Mrs", "Mr"],
    })
    for column, value in overrides.items():
        frame[column] = value
    return frame


@pytest.fixture(scope="module")
def reference_train(raw_data):
    """A pristine training frame for imputation lookups."""
    return raw_data[0]


class TestValidateManifest:
    def test_complete_manifest_passes(self):
        assert validate_manifest(compact_manifest()) == []

    def test_missing_required_columns_are_reported(self):
        frame = compact_manifest().drop(columns=["Age", "Fare"])
        problems = validate_manifest(frame)
        assert any("Age" in p and "Fare" in p for p in problems)

    def test_empty_frame_is_reported(self):
        assert validate_manifest(pd.DataFrame()) == ["Manifest contains no rows."]

    def test_non_dataframe_is_reported(self):
        problems = validate_manifest("not a table")
        assert problems and "CSV" in problems[0]

    def test_unsupported_title_is_reported(self):
        problems = validate_manifest(compact_manifest(Title=["Captain", "Mr"]))
        assert any("Unsupported Title" in p for p in problems)

    def test_unsupported_port_is_reported(self):
        problems = validate_manifest(compact_manifest(Embarked=["X", "S"]))
        assert any("Unsupported Embarked" in p for p in problems)

    def test_wholly_non_numeric_age_is_reported(self):
        problems = validate_manifest(compact_manifest(Age=["old", "young"]))
        assert any("non-numeric" in p for p in problems)


class TestNormalizeManifest:
    def test_title_supplies_the_name(self):
        frame = normalize_manifest(compact_manifest())
        assert frame["Name"].str.contains("Mrs").any()
        assert frame["Name"].str.contains("Mr").any()

    def test_existing_name_is_preserved(self):
        frame = compact_manifest()
        frame["Name"] = ["Rothschild, Mrs. Martin", "Braund, Mr. Owen"]
        normalized = normalize_manifest(frame)
        assert normalized["Name"].tolist() == ["Rothschild, Mrs. Martin", "Braund, Mr. Owen"]

    def test_absent_optional_columns_are_defaulted(self):
        normalized = normalize_manifest(compact_manifest())
        assert (normalized["Cabin"].isnull()).all()
        assert (normalized["PassengerId"].notnull()).all()
        assert (normalized["Ticket"] == "MANIFEST").all()

    def test_passenger_ids_are_generated_when_absent(self):
        normalized = normalize_manifest(compact_manifest())
        assert normalized["PassengerId"].tolist() == [1, 2]

    def test_sex_and_embarked_are_normalised(self):
        frame = compact_manifest(Sex=[" Female ", "MALE"], Embarked=[" c ", "s"])
        normalized = normalize_manifest(frame)
        assert normalized["Sex"].tolist() == ["female", "male"]
        assert normalized["Embarked"].tolist() == ["C", "S"]

    def test_input_frame_is_not_mutated(self):
        frame = compact_manifest()
        before = frame.copy()
        normalize_manifest(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_empty_frame_raises(self):
        with pytest.raises(ManifestError, match="empty"):
            normalize_manifest(pd.DataFrame())

    def test_non_dataframe_raises(self):
        with pytest.raises(ManifestError):
            normalize_manifest([1, 2, 3])

    def test_unsupported_title_raises(self):
        with pytest.raises(ManifestError, match="Unsupported Title"):
            normalize_manifest(compact_manifest(Title=["Captain", "Mr"]))


class TestPrepareManifest:
    def test_output_matches_the_model_contract(self, modeling_data, reference_train):
        feature_names = list(modeling_data[0].columns)
        aligned, engineered = prepare_manifest(compact_manifest(), feature_names, reference_train)
        assert list(aligned.columns) == feature_names
        assert len(aligned) == len(engineered) == 2

    def test_matrix_is_finite(self, modeling_data, reference_train):
        feature_names = list(modeling_data[0].columns)
        aligned, _ = prepare_manifest(compact_manifest(), feature_names, reference_train)
        assert np.isfinite(aligned.to_numpy(dtype=float)).all()

    def test_missing_age_is_imputed_from_the_training_set(self, modeling_data, reference_train):
        feature_names = list(modeling_data[0].columns)
        frame = compact_manifest(Age=[np.nan, 45.0])
        _, engineered = prepare_manifest(frame, feature_names, reference_train)
        assert not engineered["Age"].isnull().any()

    def test_kaggle_test_split_scores_end_to_end(self, modeling_data, raw_data):
        feature_names = list(modeling_data[0].columns)
        train, test = raw_data
        aligned, engineered = prepare_manifest(test, feature_names, train)
        assert len(aligned) == 418
        assert list(aligned.columns) == feature_names


class TestScoreManifest:
    def test_predictions_match_the_single_passenger_path(self, modeling_data, raw_data, repo_root):
        """Batch scoring and the simulator must agree for the same passenger."""
        import os

        import joblib

        from features import build_passenger_record

        artifact = os.path.join(repo_root, "models", "best_model.pkl")
        if not os.path.exists(artifact):
            pytest.skip("Trained model artifact not present.")

        data = joblib.load(artifact)
        model, feature_names = data["model"], data["feature_names"]

        train, test = raw_data
        aligned, _ = prepare_manifest(test.head(5), feature_names, train)
        batch_probs, _ = score_manifest(model, aligned)

        first = test.iloc[0]
        single, _ = build_passenger_record(
            feature_names,
            pclass=int(first["Pclass"]),
            sex=first["Sex"],
            age=float(first["Age"]) if pd.notnull(first["Age"]) else 28.0,
            title="Mr",
            sibsp=int(first["SibSp"]),
            parch=int(first["Parch"]),
            fare=float(first["Fare"]),
            embarked=str(first["Embarked"]),
            cabin=first["Cabin"] if pd.notnull(first["Cabin"]) else None,
        )
        single_prob = model.predict_proba(single)[0][1]
        assert batch_probs[0] == pytest.approx(single_prob, abs=1e-9)

    def test_probs_and_preds_are_consistent(self, modeling_data, raw_data, repo_root):
        import os

        import joblib

        artifact = os.path.join(repo_root, "models", "best_model.pkl")
        if not os.path.exists(artifact):
            pytest.skip("Trained model artifact not present.")

        model = joblib.load(artifact)["model"]
        feature_names = list(modeling_data[0].columns)
        train, test = raw_data
        aligned, _ = prepare_manifest(test, feature_names, train)
        probs, preds = score_manifest(model, aligned)
        assert set(np.unique(preds)) <= {0, 1}
        assert ((probs >= 0.5).astype(int) == preds).all()
        assert ((probs >= 0) & (probs <= 1)).all()


class TestSummarisePredictions:
    def test_sorted_by_descending_probability(self):
        summary = summarise_predictions([0.2, 0.9, 0.5], [0, 1, 0], [3, 1, 2])
        assert summary["SurvivalProbability"].tolist() == [0.9, 0.5, 0.2]
        assert summary["PassengerId"].tolist() == [1, 2, 3]

    def test_outcome_labels_match_predictions(self):
        summary = summarise_predictions([0.2, 0.9], [0, 1], [1, 2])
        labels = dict(zip(summary["PassengerId"], summary["Outcome"]))
        assert labels[1] == "Perished"
        assert labels[2] == "Rescued"

    def test_homogeneous_manifest_encodes_deck_correctly(self, modeling_data, reference_train):
        """Regression: a batch sharing one deck value used to lose that dummy.

        pd.get_dummies(drop_first=True) drops the only observed category, so a
        manifest of cabin-less passengers encoded Deck_U as 0 and disagreed with
        both training and the single-passenger path.
        """
        feature_names = list(modeling_data[0].columns)
        manifest = pd.DataFrame({
            "Pclass": [3, 3, 3],
            "Sex": ["male", "male", "male"],
            "Age": [22.0, 30.0, 45.0],
            "SibSp": [1, 0, 0],
            "Parch": [0, 0, 0],
            "Fare": [7.25, 8.05, 7.9],
            "Embarked": ["S", "S", "S"],
        })
        aligned, _ = prepare_manifest(manifest, feature_names, reference_train)
        assert (aligned["Deck_U"] == 1).all()
        assert (aligned["Sex_male"] == 1).all()
        assert (aligned["Title_Mr"] == 1).all()

    def test_has_kaggle_columns(self):
        summary = summarise_predictions([0.5], [0], [1])
        assert "PassengerId" in summary.columns
        assert "Survived" in summary.columns
