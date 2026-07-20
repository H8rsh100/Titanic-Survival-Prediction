# 🚢 Titanic Survival Prediction & Simulator Engine

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CV Accuracy](https://img.shields.io/badge/CV%20Accuracy-84.17%25-brightgreen?style=for-the-badge)](#-model-cross-validation-benchmark)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning pipeline and interactive AI simulator predicting passenger survival on the Titanic dataset (*Kaggle ML Competition*). Featuring rigorous domain feature engineering, Stratified 5-Fold Cross-Validation, hyperparameter optimization, and a **Soft Voting Classifier Ensemble** achieving **84.17% CV Accuracy**.

---

## 💡 Key Highlights & Features

- 🧬 **Domain-Specific Feature Engineering**:
  - **Title Extraction**: Derived high-signal social titles (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) from passenger names.
  - **Family Dynamics**: Created composite metrics (`FamilySize`, `IsAlone`, `SmallFamily`, `LargeFamily`).
  - **Cabin Deck Classification**: Extracted deck location codes (`Deck`) and handled missing values cleanly (`HasCabin`).
  - **Log-Transformed Fare**: Addressed severe right-skewness in ticket prices using $\log(1 + \text{Fare})$.
- 🛡️ **Leakage-Free Imputation**: Imputed missing `Age` values conditionally using median grouped by `Pclass` and `Title` computed strictly on training folds.
- 📊 **Stratified 5-Fold CV Benchmark**: Benchmarked Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, Support Vector Classifier, and Ensembles.
- 🔮 **Interactive Web Simulator (`app.py`)**: Built-in Streamlit web application providing real-time survival probability calculations and visual explanations.

---

## 📊 Model Cross-Validation Benchmark

All models were evaluated using **5-Fold Stratified K-Fold Cross-Validation** to prevent optimistic bias on small datasets (~891 records):

| Model | Model Family | 5-Fold Mean CV Accuracy | Std Dev |
| :--- | :--- | :---: | :---: |
| **Voting Ensemble (Soft)** 🏆 | **Ensemble (RF + GB + ET + SVC + LR)** | **84.17%** | **±0.0207** |
| Gradient Boosting (Tuned) | Tree Boosting | 84.06% | ±0.0215 |
| Random Forest (Tuned) | Bagged Trees | 83.50% | ±0.0141 |
| Logistic Regression | Linear Model | 83.05% | ±0.0170 |
| Extra Trees Classifier | Tree Ensemble | 83.16% | ±0.0036 |
| Support Vector Machine (SVC) | Kernel Model | 82.38% | ±0.0085 |

---

## 📁 Repository Structure

```
titanic-survival-prediction/
├── data/
│   ├── raw/                # train.csv, test.csv, gender_submission.csv
│   └── processed/          # Engineered train/test datasets
├── notebooks/
│   └── 01_eda_to_model.ipynb # Complete narrative EDA & ML notebook
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # Data loading, missing value imputation
│   ├── features.py         # Feature engineering & One-Hot encoding
│   ├── train.py            # Stratified K-Fold CV, hyperparameter tuning & saving
│   ├── evaluate.py         # Metric calculation & figure visualization
│   └── predict.py          # Kaggle submission CSV generator
├── models/
│   └── best_model.pkl      # Saved trained Soft Voting Classifier
├── reports/
│   └── figures/            # Generated high-resolution plots
│       ├── cv_model_comparison.png
│       ├── feature_importance.png
│       ├── confusion_matrix.png
│       └── survival_by_sex_pclass.png
├── app.py                  # Streamlit Interactive Survival Simulator App
├── submission.csv          # Kaggle-ready submission file (418 test records)
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
└── .gitignore
```

---

## ⚡ Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/H8rsh100/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
pip install -r requirements.txt
```

### 2. Run Training Pipeline
Execute full data prep, feature engineering, model tuning, and evaluation:
```bash
python src/train.py
```

### 3. Generate Kaggle Submission File
```bash
python src/predict.py
```
This produces `submission.csv` containing binary survival predictions formatted for Kaggle upload.

### 4. Launch Interactive Web App Dashboard
```bash
streamlit run app.py
```

---

## 🏆 Top Engineered Feature Importances

The most influential predictors identified by the ensemble model:

1. **Title (Mr / Miss / Mrs / Master)** — Strongest indicator of rescue prioritization.
2. **Sex (Female vs Male)** — Reflects historical "women and children first" maritime evacuation protocol.
3. **Pclass (Passenger Class)** — Socioeconomic status directly affected access to upper boat decks.
4. **Fare / Fare_Log** — Ticket cost correlates strongly with cabin deck proximity.
5. **FamilySize / IsAlone** — Large families faced coordination challenges during evacuation.

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.
