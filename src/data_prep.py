import os
import pandas as pd
import numpy as np

def load_data(data_dir="data/raw"):
    """
    Load train and test Titanic CSV datasets.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path):
        train_path = "train.csv"
    if not os.path.exists(test_path):
        test_path = "test.csv"
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def impute_missing(train_df, test_df):
    """
    Impute missing values in train and test datasets without data leakage.
    - Age: Imputed using median grouped by Title and Pclass.
    - Embarked: Imputed using mode from train.
    - Fare: Imputed using Pclass median from train.
    """
    train_clean = train_df.copy()
    test_clean = test_df.copy()
    
    # Extract temporary title for group-wise age imputation
    for df in [train_clean, test_clean]:
        df['TempTitle'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        df['TempTitle'] = df['TempTitle'].replace(title_map)
        df['TempTitle'] = df['TempTitle'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs', 'Master'] else 'Rare')
    
    # Age median map calculated ONLY on train
    age_lookup = train_clean.groupby(['Pclass', 'TempTitle'])['Age'].median()
    overall_age_median = train_clean['Age'].median()
    
    def fill_age(row):
        if pd.isnull(row['Age']):
            key = (row['Pclass'], row['TempTitle'])
            return age_lookup.get(key, overall_age_median)
        return row['Age']

    train_clean['Age'] = train_clean.apply(fill_age, axis=1)
    test_clean['Age'] = test_clean.apply(fill_age, axis=1)
    
    # Embarked mode from train
    embarked_mode = train_clean['Embarked'].mode()[0]
    train_clean['Embarked'] = train_clean['Embarked'].fillna(embarked_mode)
    test_clean['Embarked'] = test_clean['Embarked'].fillna(embarked_mode)
    
    # Fare median per Pclass from train
    fare_lookup = train_clean.groupby('Pclass')['Fare'].median()
    overall_fare_median = train_clean['Fare'].median()
    
    def fill_fare(row):
        if pd.isnull(row['Fare']):
            return fare_lookup.get(row['Pclass'], overall_fare_median)
        return row['Fare']

    train_clean['Fare'] = train_clean.apply(fill_fare, axis=1)
    test_clean['Fare'] = test_clean.apply(fill_fare, axis=1)
    
    # Cleanup temporary title column
    train_clean.drop(columns=['TempTitle'], inplace=True)
    test_clean.drop(columns=['TempTitle'], inplace=True)
    
    return train_clean, test_clean

if __name__ == "__main__":
    train, test = load_data()
    print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
    train_c, test_c = impute_missing(train, test)
    print("Missing values in Train after imputation:")
    print(train_c.isnull().sum()[train_c.isnull().sum() > 0])
