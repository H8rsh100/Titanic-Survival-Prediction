"""Generate the Kaggle submission file from the saved model artifact.

The model's feature contract is read from the pickle and enforced against the
freshly engineered matrix, so a schema drift between training and prediction
fails loudly instead of silently feeding the estimator the wrong columns.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

from data_prep import load_data, impute_missing
from features import align_to_schema, prepare_modeling_data

SUBMISSION_COLUMNS = ["PassengerId", "Survived"]
EXPECTED_TEST_ROWS = 418


def validate_submission_frame(submission_df, expected_ids=None, expected_rows=EXPECTED_TEST_ROWS):
    """
    Validate a submission frame against the Kaggle upload contract.

    Returns the list of problems found. An empty list means the file is safe to upload.
    """
    problems = []

    if list(submission_df.columns) != SUBMISSION_COLUMNS:
        problems.append(
            f"columns must be exactly {SUBMISSION_COLUMNS}, found {list(submission_df.columns)}"
        )
        return problems

    if expected_rows is not None and len(submission_df) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(submission_df)}")

    if submission_df["Survived"].isnull().any():
        problems.append("Survived column contains null values")

    if not set(submission_df["Survived"].unique()) <= {0, 1}:
        unexpected = sorted(set(submission_df["Survived"].unique()) - {0, 1})
        problems.append(f"Survived must be binary 0/1, found extra values {unexpected}")

    if submission_df["PassengerId"].duplicated().any():
        duplicates = submission_df.loc[
            submission_df["PassengerId"].duplicated(), "PassengerId"
        ].tolist()
        problems.append(f"PassengerId contains duplicates, e.g. {duplicates[:5]}")

    if expected_ids is not None and not submission_df["PassengerId"].equals(
        pd.Series(expected_ids, name="PassengerId")
    ):
        problems.append("PassengerId values or ordering do not match the raw test split")

    return problems


def generate_predictions(
    model_path="models/best_model.pkl",
    output_csv="submission.csv",
    mirror_path=None,
    expected_ids=None,
    verbose=True,
):
    """
    Generate Titanic passenger survival predictions on the Kaggle test dataset.

    Returns
    -------
    submission_df : pd.DataFrame
        Kaggle-ready frame with exactly ``PassengerId`` and ``Survived`` columns.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run `python src/train.py` first."
        )

    model_data = joblib.load(model_path)
    model = model_data["model"]
    model_name = model_data["model_name"]
    cv_score = model_data["cv_score"]
    feature_names = model_data["feature_names"]

    if verbose:
        print(f"Loaded trained model: {model_name} (CV Accuracy: {cv_score:.4f})")
        print(f"Model expects {len(feature_names)} features")

    train_df, test_df = load_data()
    train_c, test_c = impute_missing(train_df, test_df)
    X_train, y_train, X_test, train_ids, test_ids, train_eng, test_eng = prepare_modeling_data(
        train_c, test_c
    )

    # Enforce the training-time feature contract before calling the estimator.
    X_test, missing, extra = align_to_schema(X_test, feature_names)
    if verbose:
        if missing:
            print(f"Filled {len(missing)} absent dummy column(s) with zeros: {missing}")
        if extra:
            print(f"Dropped {len(extra)} column(s) absent from the contract: {extra}")

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    submission_df = pd.DataFrame({"PassengerId": test_ids, "Survived": test_preds.astype(int)})

    problems = validate_submission_frame(submission_df, expected_ids=expected_ids)
    if problems:
        raise ValueError(
            "Generated submission failed validation:\n  - " + "\n  - ".join(problems)
        )

    submission_df.to_csv(output_csv, index=False)
    if mirror_path:
        os.makedirs(os.path.dirname(mirror_path) or ".", exist_ok=True)
        submission_df.to_csv(mirror_path, index=False)

    if verbose:
        print(f"Submission validated and saved to '{output_csv}' ({len(submission_df)} rows).")
        if mirror_path:
            print(f"Mirrored to '{mirror_path}'.")
        if test_probs is not None:
            print(f"Mean predicted survival probability: {test_probs.mean():.4f}")
        print("\nFirst 10 Predictions:")
        print(submission_df.head(10).to_string(index=False))
        print("\nPredicted Class Distribution:")
        print(
            submission_df["Survived"]
            .value_counts(normalize=True)
            .rename({0: "Perished (0)", 1: "Survived (1)"})
            .to_string()
        )

    return submission_df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the Kaggle Titanic submission file.")
    parser.add_argument(
        "--model", default="models/best_model.pkl", help="Path to the saved model artifact."
    )
    parser.add_argument(
        "--output", default="submission.csv", help="Destination CSV for Kaggle upload."
    )
    parser.add_argument(
        "--mirror",
        nargs="?",
        const="data/processed/submission.csv",
        default=None,
        help="Also write a copy to this path (defaults to data/processed/submission.csv "
             "when the flag is given with no value). Off by default so the submission "
             "exists in exactly one place.",
    )
    args = parser.parse_args(argv)

    _, test_df = load_data()
    generate_predictions(
        model_path=args.model,
        output_csv=args.output,
        mirror_path=args.mirror,
        expected_ids=test_df["PassengerId"].to_numpy(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
