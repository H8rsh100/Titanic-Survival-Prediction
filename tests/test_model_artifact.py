"""Tests covering the saved model artifact contract and the training pipeline.

These tests guard the interface between ``src/train.py`` and ``src/predict.py``:
if the artifact layout or the feature contract changes, prediction silently
breaks downstream, so it is pinned here.
"""

import importlib
import os

import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_RELPATH = os.path.join("models", "best_model.pkl")


@pytest.fixture(scope="module")
def artifact(repo_root):
    path = os.path.join(repo_root, MODEL_RELPATH)
    if not os.path.exists(path):
        pytest.skip("Trained model artifact not present, run `python src/train.py` first.")
    return joblib.load(path)


class TestModulesImportCleanly:
    @pytest.mark.parametrize("module", ["data_prep", "features", "evaluate", "predict"])
    def test_module_imports(self, module):
        assert importlib.import_module(module) is not None

    def test_train_module_exposes_pipeline_entrypoints(self):
        train = importlib.import_module("train")
        assert callable(train.evaluate_models_cv)
        assert callable(train.train_and_tune_best_model)
        assert callable(train.run_pipeline)


class TestModelArtifact:
    def test_artifact_has_the_expected_keys(self, artifact):
        for key in ["model", "model_name", "cv_score", "feature_names"]:
            assert key in artifact, f"artifact is missing '{key}'"

    def test_model_exposes_supervised_predict_methods(self, artifact):
        model = artifact["model"]
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_feature_names_are_unique_and_non_empty(self, artifact):
        names = artifact["feature_names"]
        assert names
        assert len(set(names)) == len(names)

    def test_cv_score_is_a_plausible_accuracy(self, artifact):
        score = float(artifact["cv_score"])
        assert 0.5 < score <= 1.0

    def test_model_accepts_the_engineered_matrix(self, artifact, modeling_data):
        X_train, *_ = modeling_data
        assert list(X_train.columns) == list(artifact["feature_names"])
        preds = artifact["model"].predict(X_train)
        assert len(preds) == len(X_train)
        assert set(np.unique(preds)) <= {0, 1}

    def test_probabilities_are_calibrated_to_a_distribution(self, artifact, modeling_data):
        X_test = modeling_data[2]
        proba = artifact["model"].predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_model_beats_the_majority_class_baseline(self, artifact, modeling_data):
        X_train, y_train, *_ = modeling_data
        accuracy = float(np.mean(artifact["model"].predict(X_train) == y_train))
        majority = float(np.bincount(y_train).max() / len(y_train))
        assert accuracy > majority


class TestProcessedArtifacts:
    def test_engineered_train_snapshot_exists_and_is_populated(self, repo_root):
        path = os.path.join(repo_root, "data", "processed", "train_engineered.csv")
        if not os.path.exists(path):
            pytest.skip("Processed snapshot not present, run the pipeline first.")
        frame = pd.read_csv(path)
        assert len(frame) == 891
        assert "Title" in frame.columns
        assert not frame["Age"].isnull().any()
