"""Shared pytest fixtures and path setup for the Titanic test suite."""

import os
import sys

import pytest

# Force a headless plotting backend. Without this, figure tests intermittently
# fail on machines where matplotlib auto-selects TkAgg and no display is available.
import matplotlib

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DATA_RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def raw_data():
    """Return the untouched (train, test) frames straight off disk."""
    import pandas as pd

    train_path = os.path.join(DATA_RAW_DIR, "train.csv")
    test_path = os.path.join(DATA_RAW_DIR, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        pytest.skip("Raw Titanic CSVs are unavailable, skipping data backed tests.")

    return pd.read_csv(train_path), pd.read_csv(test_path)


@pytest.fixture(scope="session")
def imputed_data(raw_data):
    from data_prep import impute_missing

    return impute_missing(*raw_data)


@pytest.fixture(scope="session")
def modeling_data(imputed_data):
    from features import prepare_modeling_data

    return prepare_modeling_data(*imputed_data)
