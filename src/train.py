import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data_prep import load_data, impute_missing
from features import prepare_modeling_data
from evaluate import (
    plot_cv_comparison, plot_confusion_matrix, plot_feature_importance,
    plot_roc_curve, plot_threshold_sweep, compute_metrics
)

MODEL_DIR = "models"
REPORTS_DIR = "reports"
RANDOM_STATE = 42

def evaluate_models_cv(X_train, y_train):
    """
    Perform 5-Fold Stratified Cross-Validation across candidate baseline models.

    Returns a tidy frame with one row per model, including the individual fold
    scores so downstream reporting never has to invent a spread value.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    models = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
        'Support Vector Machine': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, C=1.0, kernel='rbf', random_state=RANDOM_STATE))])
    }
    
    cv_results = []
    
    print("\n" + "="*50)
    print(" 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK ")
    print("="*50)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        cv_results.append({
            'Model': name,
            'Mean_CV_Accuracy': mean_score,
            'Std_Dev': std_score,
            **{f'Fold{i+1}': s for i, s in enumerate(scores)}
        })
        print(f"{name:<25} | Mean Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        
    return pd.DataFrame(cv_results)

def train_and_tune_best_model(X_train, y_train, details=None):
    """
    Tune top candidate model using GridSearchCV and build an optimal Voting Ensemble.

    Parameters
    ----------
    details : dict, optional
        When provided, it is populated with the measured ensemble spread, the
        OOF metrics and the losing candidates' scores so callers can report real
        numbers. The return signature is unchanged (notebook compatible).
    """
    print("\n" + "="*50)
    print(" HYPERPARAMETER TUNING & ENSEMBLE BUILDING ")
    print("="*50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 1. GridSearch Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
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
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
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
    et = ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=RANDOM_STATE)
    svc_pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, C=1.0, kernel='rbf', random_state=RANDOM_STATE))])
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    
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
        'cv_std': float(ensemble_scores.std()),
        'feature_names': list(X_train.columns)
    }, model_file)
    print(f"\nSaved trained best model ({model_name}) to '{model_file}'")
    
    if details is not None:
        details['ensemble_scores'] = ensemble_scores
        details['ensemble_mean'] = float(ensemble_scores.mean())
        details['ensemble_std'] = float(ensemble_scores.std())
        details['rf_score'] = float(grid_rf.best_score_)
        details['rf_params'] = grid_rf.best_params_
        details['gb_score'] = float(grid_gb.best_score_)
        details['gb_params'] = grid_gb.best_params_
        details['skf'] = skf
    
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
    details = {}
    best_model, model_name, best_score = train_and_tune_best_model(X_train, y_train, details=details)
    
    # Add ensemble row using the MEASURED spread, not a hardcoded placeholder.
    ensemble_row = pd.DataFrame([{
        'Model': 'Ensemble Voting',
        'Mean_CV_Accuracy': details['ensemble_mean'],
        'Std_Dev': details['ensemble_std'],
        **{f'Fold{i+1}': s for i, s in enumerate(details['ensemble_scores'])}
    }])
    full_cv_df = pd.concat([cv_df, ensemble_row], ignore_index=True)
    full_cv_df = full_cv_df.sort_values(by='Mean_CV_Accuracy', ascending=False).reset_index(drop=True)
    plot_cv_comparison(full_cv_df)
    
    # Persist the benchmark table so the README table is reproducible.
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cv_csv = os.path.join(REPORTS_DIR, "cv_results.csv")
    full_cv_df.round(6).to_csv(cv_csv, index=False)
    print(f"\nSaved CV benchmark table to '{cv_csv}'")
    
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
    
    # Out-of-fold predictions: scoring on data the model just memorised produces a
    # confusion matrix that flatters the model and is not a generalisation estimate.
    oof_preds = cross_val_predict(best_model, X_train, y_train, cv=details['skf'], method='predict')
    oof_probs = cross_val_predict(best_model, X_train, y_train, cv=details['skf'], method='predict_proba')[:, 1]
    oof_metrics = compute_metrics(y_train, oof_preds, oof_probs)
    
    train_metrics = compute_metrics(y_train, best_model.predict(X_train))
    print("\n" + "="*50)
    print(" OUT-OF-FOLD GENERALISATION METRICS ")
    print("="*50)
    for key, value in oof_metrics.items():
        print(f"{key:<12}: {value:.4f}")
    
    plot_confusion_matrix(y_train, oof_preds, model_name=f"{model_name} (out-of-fold)")
    plot_roc_curve(y_train, oof_probs, model_name=model_name)
    plot_threshold_sweep(y_train, oof_probs)
    
    metrics_payload = {
        'selected_model': model_name,
        'cv_mean_accuracy': float(best_score),
        'cv_std_accuracy': details['ensemble_std'],
        'oof_metrics': {k: float(v) for k, v in oof_metrics.items()},
        'resubstitution_metrics': {k: float(v) for k, v in train_metrics.items()},
        'candidates': {
            'random_forest': {'cv_score': details['rf_score'], 'params': details['rf_params']},
            'gradient_boosting': {'cv_score': details['gb_score'], 'params': details['gb_params']},
        },
        'n_train': int(len(y_train)),
        'n_test': int(len(te_ids)),
    }
    metrics_file = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    print(f"Saved metrics summary to '{metrics_file}'")
    
    return best_model, X_test, te_ids

if __name__ == "__main__":
    run_pipeline()
