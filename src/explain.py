"""Per-passenger explanations for the simulator.

A bare probability is not actionable, so these helpers answer two questions the
UI can render directly:

1. Which features pushed this passenger's odds up or down, relative to the
   training-set average passenger? A linear log-odds contribution is used, which
   is exact for the logistic member of the ensemble and a faithful, clearly
   labelled approximation for the tree members.
2. Which real passengers from the training set look most like this one, and what
   happened to them.
"""

import numpy as np
import pandas as pd

# Features shown as plain numeric drivers. One-hot columns are reported by their
# parent group so the UI does not list 18 separate Title/Deck flags.
_DRIVER_GROUPS = ('Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup')


def logistic_member(estimator):
    """Return the logistic pipeline inside a VotingClassifier, or None."""
    for _, sub in getattr(estimator, "named_estimators_", {}).items():
        if sub.__class__.__name__ == "LogisticRegression":
            return sub
        if hasattr(sub, "steps"):
            final = sub.steps[-1][1]
            if final.__class__.__name__ == "LogisticRegression":
                return final
    if estimator.__class__.__name__ == "LogisticRegression":
        return estimator
    if hasattr(estimator, "steps"):
        final = estimator.steps[-1][1]
        if final.__class__.__name__ == "LogisticRegression":
            return final
    return None


def _baseline_row(X_train):
    """Mean encoding across the training set, the reference passenger.

    The one-hot columns are boolean dtype, so they must be selected explicitly
    or the baseline silently loses every indicator.
    """
    numeric = X_train.select_dtypes(include=[np.number, "bool"]).astype(float)
    return numeric.mean().to_frame().T


def feature_drivers(model, X_train, X_row, top_n=8):
    """
    Rank features by how far this passenger sits from the training average.

    Returns a tidy frame with ``Feature``, ``Value`` and ``Contribution`` sorted
    by absolute contribution.
    """
    lr = logistic_member(model)
    columns = list(X_train.columns)
    row = X_row.reindex(columns=columns).astype(float).iloc[0]
    baseline = _baseline_row(X_train)

    scaler = None
    if hasattr(model, "named_estimators_"):
        for _, sub in model.named_estimators_.items():
            if hasattr(sub, "steps") and sub.steps[0][0] == "scaler":
                scaler = sub.steps[0][1]
                break

    if lr is None:
        return pd.DataFrame(columns=["Feature", "Value", "Contribution"])

    if scaler is not None:
        row_scaled = pd.DataFrame(
            scaler.transform(row.to_frame().T), columns=scaler.get_feature_names_out()
        )
        base_scaled = pd.DataFrame(
            scaler.transform(baseline), columns=scaler.get_feature_names_out()
        )
    else:
        row_scaled, base_scaled = row.to_frame().T, baseline

    delta = (row_scaled - base_scaled).iloc[0]
    coefs = pd.Series(lr.coef_[0], index=columns).reindex(delta.index).fillna(0.0)
    raw_contribution = delta * coefs

    value_map, contribution_map = {}, {}
    for column in raw_contribution.index:
        name = _display_name(column)
        value_map[name] = value_map.get(name, 0.0) + float(row[column])
        contribution_map[name] = contribution_map.get(name, 0.0) + float(raw_contribution[column])

    result = pd.DataFrame({
        "Feature": list(value_map),
        "Value": list(value_map.values()),
        "Contribution": list(contribution_map.values()),
    })
    result = result.reindex(
        result["Contribution"].abs().sort_values(ascending=False).index
    ).head(top_n).reset_index(drop=True)
    return result


def _display_name(column):
    """Collapse a one-hot column back to the readable feature it encodes."""
    text = str(column)
    base = text.split("_")[0]
    if base in _DRIVER_GROUPS and "_" in text:
        return base
    return text


def nearest_passengers(X_train, y_train, engineered, X_row, k=8):
    """
    Find the k training passengers closest to this one.

    Returns the matching rows of ``engineered`` plus a ``Similarity`` column, so
    the UI can show real historical analogues next to the prediction.
    """
    columns = list(X_train.columns)
    row = X_row.reindex(columns=columns).astype(float).to_numpy()
    matrix = X_train[columns].astype(float).to_numpy()

    spread = matrix.std(axis=0)
    spread[spread == 0] = 1.0
    scaled_matrix = matrix / spread
    scaled_row = row / spread

    distances = np.linalg.norm(scaled_matrix - scaled_row, axis=1)
    order = np.argsort(distances)[:k]

    out = engineered.iloc[order].copy()
    out.insert(0, "Similarity", 1.0 / (1.0 + distances[order]))
    return out.reset_index(drop=True)
