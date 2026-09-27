"""Tests for the report readers that back the Streamlit analytics dashboard."""

import json
import os

import pandas as pd
import pytest

from reports import (
    available_figures, format_metric_rows, load_cv_results, load_metrics
)


class TestLoadMetrics:
    def test_returns_none_when_absent(self, tmp_path):
        assert load_metrics(str(tmp_path / "nope.json")) is None

    def test_reads_a_valid_payload(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"cv_mean_accuracy": 0.84, "oof_metrics": {"Accuracy": 0.84}}))
        payload = load_metrics(str(path))
        assert payload["cv_mean_accuracy"] == 0.84
        assert payload["oof_metrics"]["Accuracy"] == 0.84

    def test_corrupt_json_returns_none_instead_of_raising(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert load_metrics(str(path)) is None

    def test_non_object_json_returns_none(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_metrics(str(path)) is None

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text("", encoding="utf-8")
        assert load_metrics(str(path)) is None


class TestLoadCvResults:
    def test_returns_none_when_absent(self, tmp_path):
        assert load_cv_results(str(tmp_path / "nope.csv")) is None

    def test_sorts_by_accuracy_descending(self, tmp_path):
        path = tmp_path / "cv_results.csv"
        pd.DataFrame({
            "Model": ["Weak", "Best", "Middle"],
            "Mean_CV_Accuracy": [0.70, 0.90, 0.80],
        }).to_csv(path, index=False)
        frame = load_cv_results(str(path))
        assert frame["Model"].tolist() == ["Best", "Middle", "Weak"]

    def test_missing_model_column_returns_none(self, tmp_path):
        path = tmp_path / "cv_results.csv"
        pd.DataFrame({"name": ["a"], "score": [0.5]}).to_csv(path, index=False)
        assert load_cv_results(str(path)) is None

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "cv_results.csv"
        path.write_text("", encoding="utf-8")
        assert load_cv_results(str(path)) is None

    def test_preserves_fold_columns(self, tmp_path):
        path = tmp_path / "cv_results.csv"
        pd.DataFrame({
            "Model": ["A"],
            "Mean_CV_Accuracy": [0.8],
            "Std_Dev": [0.01],
            "Fold1": [0.79],
            "Fold5": [0.81],
        }).to_csv(path, index=False)
        frame = load_cv_results(str(path))
        assert "Fold1" in frame.columns and "Fold5" in frame.columns

    def test_real_artefact_is_readable(self, repo_root):
        path = os.path.join(repo_root, "reports", "cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("CV benchmark not generated yet.")
        frame = load_cv_results(path)
        assert not frame.empty
        assert {"Model", "Mean_CV_Accuracy", "Std_Dev"} <= set(frame.columns)


class TestAvailableFigures:
    def test_missing_dir_returns_empty_dict(self, tmp_path):
        assert available_figures(str(tmp_path / "nope")) == {}

    def test_only_pngs_are_listed(self, tmp_path):
        (tmp_path / "roc_pr_curves.png").write_bytes(b"\x89PNG")
        (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
        found = available_figures(str(tmp_path))
        assert set(found) == {"roc_pr_curves"}

    def test_extension_is_stripped(self, tmp_path):
        (tmp_path / "confusion_matrix.png").write_bytes(b"\x89PNG")
        found = available_figures(str(tmp_path))
        assert found["confusion_matrix"].endswith("confusion_matrix.png")

    def test_real_figures_include_the_dashboard_set(self, repo_root):
        found = available_figures(os.path.join(repo_root, "reports", "figures"))
        if not found:
            pytest.skip("Figures not generated yet.")
        assert "cv_model_comparison" in found


class TestFormatMetricRows:
    def test_none_input_gives_empty_list(self):
        assert format_metric_rows(None) == []

    def test_empty_dict_gives_empty_list(self):
        assert format_metric_rows({}) == []

    def test_orders_rows_canonically(self):
        metrics = {
            "ROC-AUC": 0.87,
            "Accuracy": 0.84,
            "Precision": 0.81,
            "Recall": 0.75,
            "F1-Score": 0.78,
        }
        labels = [label for label, _ in format_metric_rows(metrics)]
        assert labels == ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    def test_skips_absent_keys(self):
        labels = [label for label, _ in format_metric_rows({"Accuracy": 0.8, "Recall": 0.7})]
        assert labels == ["Accuracy", "Recall"]

    def test_values_are_floats(self):
        rows = format_metric_rows({"Accuracy": 1})
        assert isinstance(rows[0][1], float) and rows[0][1] == 1.0
