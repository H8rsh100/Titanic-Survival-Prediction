"""Tests asserting the generated submission file matches the Kaggle contract."""

import os

import pandas as pd
import pytest

SUBMISSION_RELPATH = "submission.csv"
EXPECTED_ROWS = 418


@pytest.fixture(scope="module")
def submission(repo_root):
    path = os.path.join(repo_root, SUBMISSION_RELPATH)
    if not os.path.exists(path):
        pytest.skip("submission.csv not present, run `python src/predict.py` first.")
    return pd.read_csv(path)


class TestSubmissionContract:
    def test_columns_are_exactly_passengerid_and_survived(self, submission):
        assert list(submission.columns) == ["PassengerId", "Survived"]

    def test_row_count_matches_the_test_split(self, submission):
        assert len(submission) == EXPECTED_ROWS

    def test_predictions_are_binary_integers(self, submission):
        assert set(submission["Survived"].unique()) <= {0, 1}
        assert pd.api.types.is_integer_dtype(submission["Survived"])

    def test_no_nulls_in_output(self, submission):
        assert not submission.isnull().any().any()

    def test_passenger_ids_are_unique(self, submission):
        assert submission["PassengerId"].is_unique

    def test_ids_align_with_the_raw_test_split_order(self, submission, data_dir):
        test_path = os.path.join(data_dir, "test.csv")
        if not os.path.exists(test_path):
            pytest.skip("Raw test split unavailable.")
        expected = pd.read_csv(test_path)["PassengerId"]
        assert submission["PassengerId"].tolist() == expected.tolist()

    def test_both_outcomes_are_represented(self, submission):
        assert submission["Survived"].nunique() == 2

    def test_predicted_survival_rate_is_plausible(self, submission):
        rate = float(submission["Survived"].mean())
        assert 0.15 < rate < 0.70
