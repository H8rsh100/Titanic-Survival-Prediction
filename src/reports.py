"""Readers for the artefacts the training pipeline persists under reports/.

Keeping these in a module (rather than inline in app.py) means the dashboard
can be unit tested without spinning up Streamlit.
"""

import json
import os

import pandas as pd

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

METRICS_FILE = os.path.join(REPORTS_DIR, "metrics.json")
CV_RESULTS_FILE = os.path.join(REPORTS_DIR, "cv_results.csv")


def load_metrics(path=METRICS_FILE):
    """Return the metrics summary, or None when the pipeline has not been run."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def load_cv_results(path=CV_RESULTS_FILE):
    """Return the per-model CV benchmark table, or None when unavailable."""
    if not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if frame.empty or "Model" not in frame.columns:
        return None
    return frame.sort_values(by="Mean_CV_Accuracy", ascending=False).reset_index(drop=True)


def available_figures(figures_dir=FIGURES_DIR):
    """Map figure name to path for every generated figure present on disk."""
    if not os.path.isdir(figures_dir):
        return {}
    return {
        name[:-4]: os.path.join(figures_dir, name)
        for name in sorted(os.listdir(figures_dir))
        if name.endswith(".png")
    }


def format_metric_rows(metrics, keys=("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")):
    """Flatten a metrics dict into ordered (label, value) pairs for rendering."""
    if not metrics:
        return []
    return [(key, float(metrics[key])) for key in keys if key in metrics]
