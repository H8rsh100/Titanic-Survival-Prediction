import os
import joblib
import pandas as pd
from data_prep import load_data, impute_missing
from features import prepare_modeling_data

def generate_predictions(model_path="models/best_model.pkl", output_csv="submission.csv"):
    """
    Generate Titanic passenger survival predictions on Kaggle test dataset.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Run train.py first!")
        
    model_data = joblib.load(model_path)
    model = model_data['model']
    model_name = model_data['model_name']
    cv_score = model_data['cv_score']
    
    print(f"Loaded trained model: {model_name} (CV Accuracy: {cv_score:.4f})")
    
    train_df, test_df = load_data()
    train_c, test_c = impute_missing(train_df, test_df)
    X_train, y_train, X_test, t_ids, te_ids, train_eng, test_eng = prepare_modeling_data(train_c, test_c)
    
    # Generate predictions
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    submission_df = pd.DataFrame({
        'PassengerId': te_ids,
        'Survived': test_preds
    })
    
    # Save submission file to root and processed folder
    submission_df.to_csv(output_csv, index=False)
    os.makedirs("data/processed", exist_ok=True)
    submission_df.to_csv("data/processed/submission.csv", index=False)
    
    print(f"Submission saved successfully to '{output_csv}' ({len(submission_df)} rows).")
    print("\nFirst 10 Predictions:")
    print(submission_df.head(10))
    print("\nPredicted Class Distribution:")
    print(submission_df['Survived'].value_counts(normalize=True).rename({0: 'Perished (0)', 1: 'Survived (1)'}))
    
    return submission_df

if __name__ == "__main__":
    generate_predictions()
