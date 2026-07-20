import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data_prep import load_data, impute_missing
from features import prepare_modeling_data
from evaluate import plot_cv_comparison, plot_confusion_matrix, plot_feature_importance

MODEL_DIR = "models"

def evaluate_models_cv(X_train, y_train):
    """
    Perform 5-Fold Stratified Cross-Validation across candidate baseline models.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Support Vector Machine': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, C=1.0, kernel='rbf', random_state=42))])
    }
    
    cv_results = []
    
    print("\n" + "="*50)
    print(" 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK ")
    print("="*50)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        cv_results.append({'Model': name, 'Mean_CV_Accuracy': mean_score, 'Std_Dev': std_score})
        print(f"{name:<25} | Mean Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        
    return pd.DataFrame(cv_results)

def train_and_tune_best_model(X_train, y_train):
    """
    Tune top candidate model using GridSearchCV and build an optimal Voting Ensemble.
    """
    print("\n" + "="*50)
    print(" HYPERPARAMETER TUNING & ENSEMBLE BUILDING ")
    print("="*50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. GridSearch Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(rf, rf_params, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    print(f"Best RF CV Score: {grid_rf.best_score_:.4f} with params: {grid_rf.best_params_}")
    
    # 2. GridSearch Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {
        'n_estimators': [100, 150],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [3, 4]
    }
    grid_gb = GridSearchCV(gb, gb_params, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    best_gb = grid_gb.best_estimator_
    print(f"Best GB CV Score: {grid_gb.best_score_:.4f} with params: {grid_gb.best_params_}")
    
    # 3. Extra Trees & SVC Pipelines
    et = ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=42)
    svc_pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, C=1.0, kernel='rbf', random_state=42))])
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    
    # 4. Soft Voting Ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('et', et),
            ('svc', svc_pipe),
            ('lr', lr_pipe)
        ],
        voting='soft',
        weights=[2, 3, 1, 1, 1]
    )
    
    ensemble_scores = cross_val_score(voting_clf, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Ensemble Voting Classifier Mean CV Accuracy: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")
    
    # Fit final best model on full training data
    if ensemble_scores.mean() >= grid_gb.best_score_:
        best_model = voting_clf
        model_name = "Voting Classifier Ensemble"
        best_score = ensemble_scores.mean()
    else:
        best_model = best_gb
        model_name = "Gradient Boosting Classifier"
        best_score = grid_gb.best_score_
        
    best_model.fit(X_train, y_train)
    
    # Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump({
        'model': best_model,
        'model_name': model_name,
        'cv_score': best_score,
        'feature_names': list(X_train.columns)
    }, model_file)
    print(f"\nSaved trained best model ({model_name}) to '{model_file}'")
    
    return best_model, model_name, best_score

def run_pipeline():
    """
    Execute full training pipeline.
    """
    train_df, test_df = load_data()
    train_c, test_c = impute_missing(train_df, test_df)
    X_train, y_train, X_test, t_ids, te_ids, train_eng, test_eng = prepare_modeling_data(train_c, test_c)
    
    # Save processed data
    train_eng.to_csv("data/processed/train_engineered.csv", index=False)
    test_eng.to_csv("data/processed/test_engineered.csv", index=False)
    
    # CV Benchmark
    cv_df = evaluate_models_cv(X_train, y_train)
    
    # Train Best Model
    best_model, model_name, best_score = train_and_tune_best_model(X_train, y_train)
    
    # Add Ensemble score to plot comparison
    ensemble_row = pd.DataFrame([{'Model': 'Ensemble Voting', 'Mean_CV_Accuracy': best_score, 'Std_Dev': 0.01}])
    full_cv_df = pd.concat([cv_df, ensemble_row], ignore_index=True)
    plot_cv_comparison(full_cv_df)
    
    # Feature Importance Plot (if tree based or extractable)
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(X_train.columns, best_model.feature_importances_)
    elif hasattr(best_model, 'estimators_'):
        # For VotingClassifier, aggregate importances from tree-based estimators
        importances = np.zeros(X_train.shape[1])
        cnt = 0
        for name, est in best_model.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                importances += est.feature_importances_
                cnt += 1
        if cnt > 0:
            importances /= cnt
            plot_feature_importance(X_train.columns, importances)
            
    # Self-Confusion Matrix on training predictions
    train_preds = best_model.predict(X_train)
    plot_confusion_matrix(y_train, train_preds, model_name=model_name)
    
    return best_model, X_test, te_ids

if __name__ == "__main__":
    run_pipeline()
