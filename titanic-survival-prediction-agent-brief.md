# Titanic Survival Prediction — Agent Build Brief

## Project Title
**Titanic Survival Prediction**

## One-liner
Predict whether a Titanic passenger survived using classification models trained on demographic and ticket features (age, sex, class, fare, etc.), following the classic Kaggle EDA → feature engineering → modeling → evaluation → submission workflow.

## Dataset
- Kaggle: "Titanic — Machine Learning from Disaster"
- Files: `train.csv`, `test.csv`, `gender_submission.csv`
- Reference guide: "Titanic: EDA to ML, Beginner" (Kaggle notebook) — use as a workflow reference, not to copy verbatim.

---

## Repo Structure
```
titanic-survivor-predictor/
├── data/
│   ├── raw/                # train.csv, test.csv (gitignored)
│   └── processed/          # cleaned/engineered datasets
├── notebooks/
│   └── 01_eda_to_model.ipynb
├── src/
│   ├── data_prep.py         # loading, cleaning, missing value handling
│   ├── features.py          # feature engineering
│   ├── train.py              # model training + cross-validation
│   ├── evaluate.py           # metrics, confusion matrix, ROC
│   └── predict.py            # generate Kaggle submission file
├── models/
│   └── best_model.pkl
├── submission.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Agent Prompt (copy-paste ready)

```
You are building a beginner-to-intermediate ML project: "Titanic Survival Prediction."

GOAL: Build an end-to-end pipeline that predicts Titanic passenger survival (binary classification) using the Kaggle Titanic dataset, and produce a Kaggle-submittable CSV.

SCOPE & WORKFLOW (in order):

1. DATA LOADING
   - Load train.csv and test.csv into pandas DataFrames.
   - Print shape, dtypes, and head() for sanity check.

2. EXPLORATORY DATA ANALYSIS (EDA)
   - Check missing values per column (Age, Cabin, Embarked, Fare).
   - Visualize survival rate by Sex, Pclass, Age bins, Embarked, SibSp/Parch.
   - Correlation heatmap of numeric features.
   - Note class imbalance in the target (Survived).

3. MISSING VALUE HANDLING
   - Age: impute using median grouped by Pclass + Sex (not global median).
   - Embarked: impute with mode (only 2 missing rows).
   - Cabin: mostly missing — engineer a binary "HasCabin" feature instead of imputing directly, or extract deck letter where available.
   - Fare: impute single missing test-set value with median grouped by Pclass.

4. FEATURE ENGINEERING
   - Title: extract from Name (Mr, Mrs, Miss, Master, Rare) — this is one of the highest-signal engineered features in this dataset.
   - FamilySize = SibSp + Parch + 1
   - IsAlone = 1 if FamilySize == 1 else 0
   - AgeBin / FareBin: bucket into categorical bins.
   - Ticket/Cabin prefix extraction (optional, lower priority).
   - Drop: PassengerId, Name (after title extraction), Ticket, Cabin (after HasCabin extracted).

5. ENCODING
   - Label/one-hot encode: Sex, Embarked, Title.
   - Scale numeric features (Age, Fare) if using models sensitive to scale (skip for tree-based models).

6. MODELING
   - Baseline: Logistic Regression.
   - Compare against: Random Forest, Gradient Boosting (XGBoost or sklearn's GradientBoostingClassifier), SVM.
   - Use Stratified K-Fold cross-validation (5 folds) — NOT a single train/test split, since dataset is small (~891 rows).
   - Hyperparameter tuning via GridSearchCV or RandomizedSearchCV on top 1-2 candidate models.

7. EVALUATION
   - Metrics: accuracy, precision, recall, F1, ROC-AUC.
   - Confusion matrix visualization.
   - Feature importance plot (for tree-based models) or coefficient plot (for logistic regression).
   - Compare CV scores across models in a summary table.

8. PREDICTION & SUBMISSION
   - Retrain best model on full training data.
   - Predict on test.csv.
   - Output submission.csv with columns: PassengerId, Survived — matching Kaggle's required format exactly.

9. DELIVERABLES
   - notebooks/01_eda_to_model.ipynb — full narrative notebook (EDA plots, reasoning, final model).
   - src/ — modular scripts mirroring the notebook logic (for reuse/testing, not just notebook-only code).
   - README.md — problem statement, approach summary, final CV accuracy, what was tried, what worked/didn't.
   - submission.csv — ready for Kaggle upload.

CONSTRAINTS:
- Don't leak test-set information into training (fit imputers/encoders/scalers on train only, then transform test).
- Keep the notebook readable: markdown cells explaining each step's reasoning, not just code blocks.
- Target CV accuracy: aim for 80%+ (competitive beginner benchmark on this dataset is ~78-83%).
- Use scikit-learn as the primary library; XGBoost optional but recommended for the boosted model comparison.

OUTPUT: Full repo matching the structure above, with working code, a populated notebook, and a submission-ready CSV.
```

---

## Notes for You (not the agent)
- This is a well-trodden dataset — the value-add is in doing feature engineering (especially Title extraction) *carefully* and documenting your reasoning in the README, since that's what separates a copy-paste solution from something you can actually talk about in an interview.
- Push CV score comparisons (not just final accuracy) into the README — shows you understand model selection isn't just "pick the best number."
- Once done, submit to the live Kaggle leaderboard for the bonus — worth doing since it's free signal on how your pipeline generalizes.
