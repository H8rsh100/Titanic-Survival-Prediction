"""Batch (manifest) scoring helpers for the Streamlit app.

The single-passenger simulator is fine for exploring one scenario, but the real
workflow is scoring a whole passenger list. These helpers let a manifest be
encoded through exactly the same path the model was trained on, so batch and
single scoring cannot disagree.
"""

import numpy as np
import pandas as pd

from data_prep import impute_missing
from features import (
    FEATURE_CATEGORIES, RAW_DROP_COLUMNS, align_to_schema, encode_records, extract_features
)

# Columns extract_features needs, with the value used when a manifest omits them.
# Title defaults to 'Mr' because it is by far the most common title, and a manifest
# with no title at all should not silently land in the 'Rare' bucket.
_MANIFEST_DEFAULTS = {
    'PassengerId': None,
    'Name': None,
    'Pclass': 3,
    'Sex': 'male',
    'Age': np.nan,
    'SibSp': 0,
    'Parch': 0,
    'Ticket': 'MANIFEST',
    'Fare': np.nan,
    'Cabin': None,
    'Embarked': 'S',
    'Title': 'Mr',
}

REQUIRED_MANIFEST_COLUMNS = ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked')
OPTIONAL_MANIFEST_COLUMNS = ('PassengerId', 'Title', 'Cabin')


class ManifestError(ValueError):
    """Raised when an uploaded manifest cannot be scored."""


def _synthesise_name(title):
    """Build a Name that extract_features will parse back into the given title."""
    token = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master', 'Rare': 'Sir'}
    if title not in token:
        raise ManifestError(
            f"Unsupported Title '{title}'. Expected one of {sorted(token)}."
        )
    return f"Manifest, {token[title]}. Passenger"


def normalize_manifest(manifest_df):
    """
    Coerce an uploaded manifest into the raw column layout the pipeline expects.

    A ``Title`` column is accepted in place of ``Name`` (most people building a
    passenger list have a title, not a full name), and any column that is absent
    entirely falls back to a documented default.

    Returns a new frame; the caller's frame is never mutated.
    """
    if not isinstance(manifest_df, pd.DataFrame):
        raise ManifestError("Manifest must be a pandas DataFrame.")
    if manifest_df.empty:
        raise ManifestError("Manifest is empty, there are no passengers to score.")

    frame = manifest_df.copy()

    # A Title column is accepted in place of Name, so keep it and synthesise a
    # Name that extract_features can parse back into that title.
    if 'Title' not in frame.columns:
        frame['Title'] = 'Mr'
    frame['Title'] = frame['Title'].fillna('Mr').astype(str)

    for column, default in _MANIFEST_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
        elif default is not None and column not in ('Name', 'Title', 'PassengerId'):
            frame[column] = frame[column].fillna(default)

    # Build or repair Name. Rows that arrive with a real Name keep it; the rest
    # get one synthesised from their Title.
    synth_names = [_synthesise_name(title) for title in frame['Title']]
    if 'Name' in manifest_df.columns:
        supplied = manifest_df['Name']
        frame['Name'] = [
            value if pd.notnull(value) else synth
            for value, synth in zip(supplied, synth_names)
        ]
    else:
        frame['Name'] = synth_names

    # extract_features needs a string Name for its regex.
    frame['Name'] = frame['Name'].astype(str)
    frame['Sex'] = frame['Sex'].astype(str).str.strip().str.lower()
    frame['Embarked'] = frame['Embarked'].astype(str).str.strip().str.upper().str[:1]

    if frame['PassengerId'].isnull().all():
        frame['PassengerId'] = np.arange(1, len(frame) + 1)

    return frame


def validate_manifest(manifest_df):
    """Return a list of human readable problems with an uploaded manifest."""
    problems = []
    if not isinstance(manifest_df, pd.DataFrame):
        return ["Manifest must be a CSV that parses into a table."]
    if manifest_df.empty:
        return ["Manifest contains no rows."]

    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in manifest_df.columns]
    if missing:
        problems.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Optional extras: {', '.join(OPTIONAL_MANIFEST_COLUMNS)}."
        )
        return problems

    if 'Title' in manifest_df.columns and 'Name' not in manifest_df.columns:
        unknown = sorted(set(manifest_df['Title'].dropna().astype(str)) -
                         set(FEATURE_CATEGORIES['Title']))
        if unknown:
            problems.append(
                f"Unsupported Title value(s): {unknown}. "
                f"Allowed: {sorted(FEATURE_CATEGORIES['Title'])}."
            )

    if 'Embarked' in manifest_df.columns:
        unknown_ports = sorted(set(manifest_df['Embarked'].dropna().astype(str)) - {'S', 'C', 'Q'})
        if unknown_ports:
            problems.append(f"Unsupported Embarked value(s): {unknown_ports}. Allowed: C, Q, S.")

    for column in ('Age', 'Fare'):
        if column in manifest_df.columns:
            coerced = pd.to_numeric(manifest_df[column], errors='coerce')
            if coerced.notnull().sum() == 0:
                problems.append(f"Column '{column}' is entirely non-numeric.")

    if problems:
        return problems
    return []


def prepare_manifest(manifest_df, feature_names, reference_train_df):
    """
    Encode a manifest into the model's feature space.

    Missing values are imputed using medians derived from ``reference_train_df``
    only, never from the manifest itself.

    Returns
    -------
    aligned : pd.DataFrame
        Encoded matrix with exactly ``feature_names`` columns.
    engineered : pd.DataFrame
        Human readable engineered frame for display.
    """
    frame = normalize_manifest(manifest_df)
    _, frame = impute_missing(reference_train_df, frame)

    engineered = extract_features(frame)
    record = engineered.drop(columns=[c for c in RAW_DROP_COLUMNS if c in engineered.columns])

    # Shared encoder, so a batch can never encode differently from a single
    # passenger or from training.
    encoded = encode_records(record)
    aligned, _, _ = align_to_schema(encoded, feature_names)

    return aligned, engineered


def score_manifest(model, aligned_matrix):
    """Return (survival probabilities, binary predictions) for an encoded manifest."""
    probs = model.predict_proba(aligned_matrix)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def summarise_predictions(probs, preds, passenger_ids):
    """Build the result table shown in the UI, sorted by likelihood."""
    summary = pd.DataFrame({
        'PassengerId': list(passenger_ids),
        'SurvivalProbability': np.asarray(probs, dtype=float),
        'Survived': np.asarray(preds, dtype=int),
    })
    summary['Outcome'] = summary['Survived'].map({1: 'Rescued', 0: 'Perished'})
    return summary.sort_values('SurvivalProbability', ascending=False).reset_index(drop=True)
