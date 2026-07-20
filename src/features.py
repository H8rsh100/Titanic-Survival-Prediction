import pandas as pd
import numpy as np

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
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: str(x)[0].upper())
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
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
    train_features = train_eng.drop(columns=[c for c in drop_cols if c in train_eng.columns])
    test_features = test_eng.drop(columns=[c for c in drop_cols if c in test_eng.columns])
    
    # Categorical One-Hot Encoding
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']
    
    combined = pd.concat([train_features, test_features], axis=0)
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    X_train = combined_encoded.iloc[:len(train_df)].copy()
    X_test = combined_encoded.iloc[len(train_df):].copy()
    
    return X_train, y_train, X_test, train_ids, test_ids, train_eng, test_eng

if __name__ == "__main__":
    from data_prep import load_data, impute_missing
    train, test = load_data()
    train_c, test_c = impute_missing(train, test)
    X_train, y_train, X_test, t_ids, te_ids, train_e, test_e = prepare_modeling_data(train_c, test_c)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print("Features extracted:", list(X_train.columns))
