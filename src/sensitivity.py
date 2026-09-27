"""What-if analysis for the simulator.

Sweeping a single input while holding the rest fixed answers the question the
probability badge cannot: how much would this choice actually change the
outcome, and where does the model stop caring?
"""

import numpy as np
import pandas as pd

from features import build_passenger_record

# Sweeps offered in the UI. Each entry is (label, attribute, ordered values).
SWEEPS = {
    "Fare": ("fare", [0.0, 7.25, 15.0, 30.0, 60.0, 100.0, 200.0, 500.0]),
    "Age": ("age", [1.0, 6.0, 12.0, 18.0, 30.0, 45.0, 60.0, 75.0]),
    "Pclass": ("pclass", [1, 2, 3]),
    "SibSp": ("sibsp", [0, 1, 2, 3, 4, 6, 8]),
    "Parch": ("parch", [0, 1, 2, 3, 4, 6]),
    "Embarked": ("embarked", ["S", "C", "Q"]),
    "Sex": ("sex", ["male", "female"]),
}


def sensitivity_curve(model, feature_names, base_kwargs, sweep="Fare"):
    """
    Score ``base_kwargs`` across every value of one attribute.

    Returns a frame of ``Value`` and ``SurvivalProbability`` plus the change
    from the base case, so the UI can annotate the turning point.
    """
    if sweep not in SWEEPS:
        raise ValueError(f"Unknown sweep: {sweep}. Choose from {sorted(SWEEPS)}")

    attribute, values = SWEEPS[sweep]

    records, probs = [], []
    for value in values:
        kwargs = dict(base_kwargs)
        kwargs[attribute] = value
        row, _ = build_passenger_record(feature_names, **kwargs)
        records.append(row)
        probs.append(float(model.predict_proba(row)[0, 1]))

    base_prob = float(model.predict_proba(
        build_passenger_record(feature_names, **base_kwargs)[0]
    )[0, 1])

    return pd.DataFrame({
        "Value": values,
        "SurvivalProbability": probs,
        "Delta": np.array(probs) - base_prob,
    }), base_prob


def classification_threshold():
    """The 0.5 decision point implied by the probability gauge."""
    return 0.5
