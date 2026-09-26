"""Tests for feature-schema alignment and Kaggle submission validation."""

import os

import numpy as np
import pandas as pd
import pytest

from features import FeatureSchemaMismatch, align_to_schema, prepare_modeling_data
from predict import SUBMISSION_COLUMNS, generate_predictions, validate_submission_frame


def good_submission():
    return pd.DataFrame(
        {"PassengerId": np.arange(892, 892 + 418), "Survived": np.zeros(418, dtype=int)}
    )


class TestAlignToSchema:
    def test_matching_schema_is_unchanged(self, modeling_data):
        X_test = modeling_data[2]
        names = list(X_test.columns)
        aligned, missing, extra = align_to_schema(X_test, names)
        assert list(aligned.columns) == names
        assert missing == [] and extra == []
        pd.testing.assert_frame_equal(aligned, X_test)

    def test_column_order_follows_the_contract(self, modeling_data):
        X_test = modeling_data[2]
        names = list(X_test.columns)
        shuffled = list(reversed(names))
        aligned, _, _ = align_to_schema(X_test, shuffled)
        assert list(aligned.columns) == shuffled

    def test_absent_dummy_column_is_zero_filled(self, modeling_data):
        X_test = modeling_data[2]
        names = list(X_test.columns) + ["Title_Nonexistent"]
        aligned, missing, extra = align_to_schema(X_test, names)
        assert missing == ["Title_Nonexistent"]
        assert (aligned["Title_Nonexistent"] == 0).all()

    def test_unknown_extra_columns_are_dropped(self, modeling_data):
        X_test = modeling_data[2]
        polluted = X_test.copy()
        polluted["MysteryFeature"] = 1
        names = list(X_test.columns)
        aligned, missing, extra = align_to_schema(polluted, names)
        assert extra == ["MysteryFeature"]
        assert "MysteryFeature" not in aligned.columns

    def test_missing_numeric_column_raises(self, modeling_data):
        X_test = modeling_data[2]
        names = list(X_test.columns) + ["FarePerPerson"]
        with pytest.raises(FeatureSchemaMismatch, match="FarePerPerson"):
            align_to_schema(X_test, names)

    def test_non_strict_mode_tolerates_missing_numeric(self, modeling_data):
        X_test = modeling_data[2]
        names = list(X_test.columns) + ["FarePerPerson"]
        aligned, missing, _ = align_to_schema(X_test, names, strict=False)
        assert missing == ["FarePerPerson"]
        assert (aligned["FarePerPerson"] == 0).all()

    def test_alignment_preserves_row_count_and_row_order(self, modeling_data):
        X_test = modeling_data[2]
        names = list(X_test.columns) + ["Deck_Z"]
        aligned, _, _ = align_to_schema(X_test, names)
        assert len(aligned) == len(X_test)
        np.testing.assert_allclose(aligned["Age"].to_numpy(), X_test["Age"].to_numpy())


class TestValidateSubmissionFrame:
    def test_valid_frame_passes(self):
        assert validate_submission_frame(good_submission()) == []

    def test_wrong_columns_are_rejected(self):
        frame = good_submission().rename(columns={"Survived": "survived"})
        problems = validate_submission_frame(frame)
        assert any("columns" in p for p in problems)

    def test_wrong_row_count_is_rejected(self):
        problems = validate_submission_frame(good_submission().head(10))
        assert any("expected 418 rows" in p for p in problems)

    def test_null_predictions_are_rejected(self):
        frame = good_submission()
        frame.loc[3, "Survived"] = np.nan
        assert any("null" in p for p in validate_submission_frame(frame))

    def test_non_binary_predictions_are_rejected(self):
        frame = good_submission()
        frame.loc[0, "Survived"] = 2
        assert any("binary" in p for p in validate_submission_frame(frame))

    def test_duplicate_ids_are_rejected(self):
        frame = good_submission()
        frame.loc[1, "PassengerId"] = frame.loc[0, "PassengerId"]
        assert any("duplicates" in p for p in validate_submission_frame(frame))

    def test_misordered_ids_are_rejected(self):
        frame = good_submission()
        frame["PassengerId"] = frame["PassengerId"].iloc[::-1].to_numpy()
        expected = good_submission()["PassengerId"].to_numpy()
        assert any("ordering" in p for p in validate_submission_frame(frame, expected_ids=expected))

    def test_matching_ids_are_accepted(self):
        frame = good_submission()
        assert validate_submission_frame(frame, expected_ids=frame["PassengerId"].to_numpy()) == []

    def test_column_check_short_circuits_before_row_checks(self):
        frame = good_submission().drop(columns=["Survived"])
        problems = validate_submission_frame(frame)
        assert len(problems) == 1


class TestGeneratePredictions:
    def test_missing_model_raises_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            generate_predictions(model_path=str(tmp_path / "nope.pkl"), verbose=False)

    def test_end_to_end_output_satisfies_the_contract(self, repo_root, tmp_path, raw_data):
        model_path = os.path.join(repo_root, "models", "best_model.pkl")
        if not os.path.exists(model_path):
            pytest.skip("Trained model artifact not present, run `python src/train.py` first.")

        out = tmp_path / "sub.csv"
        submission = generate_predictions(
            model_path=model_path,
            output_csv=str(out),
            mirror_path=None,
            expected_ids=raw_data[1]["PassengerId"].to_numpy(),
            verbose=False,
        )

        assert list(submission.columns) == SUBMISSION_COLUMNS
        assert len(submission) == 418
        assert validate_submission_frame(submission) == []
        assert out.exists()
        pd.testing.assert_frame_equal(pd.read_csv(out), submission)

    def test_ids_are_rejected_when_they_do_not_match_the_test_split(self, repo_root, tmp_path):
        model_path = os.path.join(repo_root, "models", "best_model.pkl")
        if not os.path.exists(model_path):
            pytest.skip("Trained model artifact not present, run `python src/train.py` first.")

        with pytest.raises(ValueError, match="failed validation"):
            generate_predictions(
                model_path=model_path,
                output_csv=str(tmp_path / "bad.csv"),
                mirror_path=None,
                expected_ids=np.arange(1, 419),
                verbose=False,
            )


class TestFeatureEncodingContract:
    def test_dummy_columns_cover_every_categorical(self, modeling_data):
        from features import CATEGORICAL_COLUMNS

        X_train, *_ = modeling_data
        for column in CATEGORICAL_COLUMNS:
            assert any(name.startswith(f"{column}_") for name in X_train.columns)

    def test_prepare_modeling_data_is_deterministic(self, imputed_data):
        first = prepare_modeling_data(*imputed_data)
        second = prepare_modeling_data(*imputed_data)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[2], second[2])
