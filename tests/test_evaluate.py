"""Tests for metric computation and the evaluation figure helpers."""

import os

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    compute_metrics, plot_confusion_matrix, plot_cv_comparison, plot_feature_importance,
    plot_roc_curve, plot_threshold_sweep
)


@pytest.fixture
def toy_predictions():
    rng = np.random.default_rng(0)
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.05, 0.20, 0.90, 0.60, 0.40, 0.30, 0.80, 0.45, 0.55, 0.10])
    return y_true, y_prob, rng


class TestComputeMetrics:
    def test_metrics_without_probabilities(self):
        metrics = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])
        assert set(metrics) == {"Accuracy", "Precision", "Recall", "F1-Score"}
        assert metrics["Accuracy"] == pytest.approx(0.75)

    def test_roc_auc_is_added_when_probabilities_are_supplied(self, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        metrics = compute_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
        assert "ROC-AUC" in metrics
        assert 0.0 <= metrics["ROC-AUC"] <= 1.0

    def test_perfect_and_inverted_separations(self):
        y_true = [0, 0, 1, 1]
        assert compute_metrics(y_true, y_true, [0.1, 0.2, 0.8, 0.9])["ROC-AUC"] == pytest.approx(1.0)
        assert compute_metrics(y_true, y_true, [0.9, 0.8, 0.2, 0.1])["ROC-AUC"] == pytest.approx(0.0)

    def test_all_values_are_finite(self, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        metrics = compute_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
        assert all(np.isfinite(v) for v in metrics.values())


class TestFigureHelpers:
    def test_confusion_matrix_is_written(self, tmp_path, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        path = plot_confusion_matrix(
            y_true, (y_prob >= 0.5).astype(int), save_dir=str(tmp_path)
        )
        assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_roc_pr_curve_is_written(self, tmp_path, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        path = plot_roc_curve(y_true, y_prob, save_dir=str(tmp_path))
        assert os.path.exists(path) and path.endswith("roc_pr_curves.png")

    def test_threshold_sweep_is_written(self, tmp_path, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        path = plot_threshold_sweep(y_true, y_prob, save_dir=str(tmp_path))
        assert os.path.exists(path) and path.endswith("threshold_sweep.png")

    def test_cv_comparison_accepts_fold_columns(self, tmp_path):
        frame = pd.DataFrame({
            "Model": ["A", "B"],
            "Mean_CV_Accuracy": [0.80, 0.85],
            "Std_Dev": [0.01, 0.02],
            "Fold1": [0.79, 0.84],
        })
        path = plot_cv_comparison(frame, save_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_feature_importance_is_written(self, tmp_path):
        names = ["a", "b", "c"]
        path = plot_feature_importance(names, [0.1, 0.7, 0.2], save_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_save_dir_is_created_if_absent(self, tmp_path, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        target = tmp_path / "deep" / "nested"
        plot_roc_curve(y_true, y_prob, save_dir=str(target))
        assert target.is_dir()


class TestThresholdSweepLogic:
    """Regression cover for the argmin-vs-absolute-argmin bug that reported the
    0.05 threshold as the 0.50 default."""

    def _sweep_frame(self, y_true, y_prob, n_points=101):
        from sklearn.metrics import accuracy_score, f1_score

        thresholds = np.linspace(0.05, 0.95, n_points)
        rows = []
        for threshold in thresholds:
            preds = (y_prob >= threshold).astype(int)
            rows.append({
                "Threshold": threshold,
                "Accuracy": accuracy_score(y_true, preds),
                "F1": f1_score(y_true, preds, zero_division=0),
            })
        return pd.DataFrame(rows), thresholds

    def test_nearest_threshold_to_half_is_really_half(self):
        _, thresholds = self._sweep_frame([0, 1], [0.1, 0.9])
        index = int(np.abs(thresholds - 0.5).argmin())
        assert thresholds[index] == pytest.approx(0.5)
        assert index == 50

    def test_plain_argmin_would_have_picked_the_wrong_row(self):
        _, thresholds = self._sweep_frame([0, 1], [0.1, 0.9])
        assert int((thresholds - 0.5).argmin()) == 0
        assert thresholds[int((thresholds - 0.5).argmin())] == pytest.approx(0.05)

    def test_default_row_accuracy_matches_a_direct_computation(self, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        sweep, thresholds = self._sweep_frame(y_true, y_prob)
        index = int(np.abs(thresholds - 0.5).argmin())
        direct = float(np.mean((y_prob >= thresholds[index]).astype(int) == y_true))
        assert sweep.iloc[index]["Accuracy"] == pytest.approx(direct)

    def test_monotone_threshold_pushes_recall_down(self, toy_predictions):
        y_true, y_prob, _ = toy_predictions
        sweep, _ = self._sweep_frame(y_true, y_prob)
        preds_low = (y_prob >= 0.05).astype(int)
        preds_high = (y_prob >= 0.95).astype(int)
        assert preds_low.sum() > preds_high.sum()
