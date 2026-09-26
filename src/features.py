import pandas as pd
import numpy as np

CATEGORICAL_COLUMNS = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']

# Full category universe for each encoded column, in the order pd.get_dummies
# produces with drop_first=True. The FIRST level is the dropped reference
# category, every subsequent level becomes a `{column}_{level}` dummy.
# These are domain-stable (a deck is a letter A-G, a title collapses to five
# groups), so the same universe holds for single-row scoring.
FEATURE_CATEGORIES = {
    'Sex': ('female', 'male'),
    'Embarked': ('C', 'Q', 'S'),
    'Title': ('Master', 'Miss', 'Mr', 'Mrs', 'Rare'),
    'Deck': ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'U'),
    'AgeGroup': ('Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'),
}

# Columns that are dropped before the one-hot step and never reach the model.
RAW_DROP_COLUMNS = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']


class FeatureSchemaMismatch(ValueError):
    """Raised when a feature matrix cannot be reconciled with a model's contract."""

def extract_features(df):
    """
    Extract domain-specific engineered features from raw Titanic passenger records.
    """
    df = df.copy()
    
    # 1. Title Extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    df['Title'] = df['Title'].replace(title_map)
    df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs', 'Master'] else 'Rare')
    
    # 2. Family Features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
    df['LargeFamily'] = (df['FamilySize'] > 4).astype(int)
    
    # 3. Cabin & Deck Features
    # Normalise blank/whitespace-only cabin labels to "no cabin recorded" so that
    # str[0] below can never index into an empty string.
    cabin = df['Cabin'].astype('string').str.strip()
    has_cabin = (cabin.notna()) & (cabin.str.len() > 0)
    df['HasCabin'] = has_cabin.astype(int)
    df['Deck'] = cabin.where(has_cabin, 'U').str[0].str.upper().astype(object)
    df['Deck'] = df['Deck'].replace(['T'], 'U')  # Group ultra-rare deck T with U
    
    # 4. Age Binning
    bins = [0, 12, 18, 35, 60, 120]
    labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # 5. Log Fare Transformation
    df['Fare_Log'] = np.log1p(df['Fare'])
    
    return df

def prepare_modeling_data(train_df, test_df):
    """
    Apply feature engineering and one-hot encoding on train and test datasets.
    Returns feature matrices X_train, X_test, target y_train, and PassengerIds.
    """
    train_eng = extract_features(train_df)
    test_eng = extract_features(test_df)
    
    # Target variable and ID tracking
    y_train = train_eng['Survived'].values
    train_ids = train_eng['PassengerId'].values
    test_ids = test_eng['PassengerId'].values
    
    # Drop identifier & text columns
    drop_cols = RAW_DROP_COLUMNS
    train_features = train_eng.drop(columns=[c for c in drop_cols if c in train_eng.columns])
    test_features = test_eng.drop(columns=[c for c in drop_cols if c in test_eng.columns])
    
    # Categorical One-Hot Encoding
    cat_cols = CATEGORICAL_COLUMNS
    
    combined = pd.concat([train_features, test_features], axis=0)
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    X_train = combined_encoded.iloc[:len(train_df)].copy()
    X_test = combined_encoded.iloc[len(train_df):].copy()
    
    return X_train, y_train, X_test, train_ids, test_ids, train_eng, test_eng


def _is_dummy_column(column):
    """Return True when the column looks like a one-hot of a known categorical."""
    return any(
        column.startswith(f"{cat}_") or column == f"{cat}"
        for cat in CATEGORICAL_COLUMNS
    )


def align_to_schema(X, feature_names, strict=True):
    """
    Reconcile a feature matrix with the column contract recorded in a model artifact.

    One-hot columns are derived from observed categories, so a matrix built from a
    different slice of data can be missing indicator columns that the model still
    expects. Absent dummy columns are safe to fill with 0 (the category simply did
    not occur); a missing numeric column is a real defect and raises instead.

    Returns
    -------
    aligned : pd.DataFrame
        Matrix with exactly ``feature_names`` columns, in that order.
    missing : list[str]
        Dummy columns that were filled with zeros.
    extra : list[str]
        Columns present in ``X`` but absent from the contract (dropped).
    """
    feature_names = list(feature_names)
    missing = [name for name in feature_names if name not in X.columns]
    extra = [column for column in X.columns if column not in feature_names]

    unrecoverable = [name for name in missing if not _is_dummy_column(name)]
    if unrecoverable and strict:
        raise FeatureSchemaMismatch(
            "Feature matrix is missing required non-encoded columns: "
            f"{unrecoverable}. Refit the pipeline or align the training schema."
        )

    aligned = X.reindex(columns=feature_names)
    for name in missing:
        aligned[name] = aligned[name].fillna(0)

    return aligned, missing, extra


def encode_single_record(record_df):
    """
    One-hot encode a one-row engineered frame against the full category universe.

    ``pd.get_dummies(..., drop_first=True)`` derives its columns from the
    categories present in the data. On a single row that leaves exactly one
    category, which drop_first then discards, so every dummy for that column
    comes out zero. Scoring one passenger would therefore erase the signal
    entirely (a "male" and a "female" record would encode identically).
    Encoding against the explicit universe instead keeps the reference category
    all-zero and sets the one matching dummy.
    """
    if len(record_df) != 1:
        raise ValueError(f"encode_single_record expects exactly 1 row, got {len(record_df)}")

    encoded = {}
    for column in record_df.columns:
        value = record_df[column].iloc[0]
        if column in CATEGORICAL_COLUMNS:
            value = str(value)
            for level in FEATURE_CATEGORIES[column][1:]:
                encoded[f"{column}_{level}"] = 1 if value == level else 0
        else:
            encoded[column] = value

    return pd.DataFrame([encoded])


# Representative name token used to synthesise a Name for each title group.
_TITLE_NAME_TOKEN = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master', 'Rare': 'Sir'}


def build_passenger_record(feature_names, pclass=3, sex='male', age=28.0, title='Mr',
                           sibsp=0, parch=0, fare=32.2, embarked='S', cabin=None):
    """
    Build a single engineered, one-hot encoded passenger row for ad-hoc scoring.

    The interactive simulator used to hand-assemble the feature vector, which
    silently skipped the ``AgeGroup`` dummies and allowed impossible combinations
    such as ``HasCabin=0`` paired with a named deck. Routing through
    ``extract_features`` makes the simulator produce exactly the same encoding the
    model was trained on.

    Parameters
    ----------
    feature_names : sequence of str
        Column contract recorded in the model artifact.
    cabin : str or None
        Cabin label. ``None`` means no cabin was recorded, which yields
        ``HasCabin=0`` and ``Deck='U'`` by construction.

    Returns
    -------
    aligned : pd.DataFrame
        One-row matrix matching ``feature_names``.
    engineered : pd.DataFrame
        One-row human readable engineered frame (useful for display/debugging).
    """
    if title not in _TITLE_NAME_TOKEN:
        raise ValueError(
            f"Unsupported title '{title}'. Expected one of {sorted(_TITLE_NAME_TOKEN)}."
        )
    if embarked not in ('S', 'C', 'Q'):
        raise ValueError(f"Unsupported embarkation port '{embarked}'. Expected S, C or Q.")

    raw = pd.DataFrame([{
        'PassengerId': 0,
        'Survived': 0,
        'Pclass': pclass,
        'Name': f"Simulated, {_TITLE_NAME_TOKEN[title]}. Passenger",
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': 'SIM-000',
        'Fare': fare,
        'Cabin': cabin,
        'Embarked': embarked,
    }])

    engineered = extract_features(raw)
    record = engineered.drop(columns=[c for c in RAW_DROP_COLUMNS if c in engineered.columns])
    encoded = encode_single_record(record)
    aligned, _, _ = align_to_schema(encoded, feature_names)

    return aligned, engineered

if __name__ == "__main__":
    from data_prep import load_data, impute_missing
    train, test = load_data()
    train_c, test_c = impute_missing(train, test)
    X_train, y_train, X_test, t_ids, te_ids, train_e, test_e = prepare_modeling_data(train_c, test_c)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print("Features extracted:", list(X_train.columns))
