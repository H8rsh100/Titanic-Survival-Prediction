import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Titanic Survival Predictor — AI Simulator",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_pipeline():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        return None, None
    data = joblib.load(model_path)
    return data['model'], data['feature_names']

st.markdown('<div class="main-header">🚢 Titanic Survival Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Disaster Survival Simulator & Kaggle ML Pipeline Dashboard</div>', unsafe_allow_html=True)

model, feature_names = load_model_and_pipeline()

if model is None:
    st.error("Trained model artifact not found. Please run `python src/train.py` first!")
    st.stop()

tabs = st.tabs(["🔮 Passenger Survival Simulator", "📊 Benchmark & Feature Insights", "📄 Model Metadata"])

with tabs[0]:
    st.subheader("Interactive Passenger Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pclass = st.selectbox("Ticket Class (Pclass)", options=[1, 2, 3], index=2, help="1 = First Class, 2 = Second Class, 3 = Third Class")
        sex = st.selectbox("Sex", options=["female", "male"], index=1)
        age = st.slider("Age", min_value=0, max_value=80, value=28, step=1)
        title = st.selectbox("Title", options=["Mr", "Mrs", "Miss", "Master", "Rare"], index=0)

    with col2:
        sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
        parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0)
        fare = st.slider("Ticket Fare ($)", min_value=0.0, max_value=500.0, value=32.2, step=1.0)
        
    with col3:
        embarked = st.selectbox("Port of Embarkation", options=["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"], index=0)
        embarked_code = embarked.split(" ")[0]
        has_cabin = st.selectbox("Has Recorded Cabin?", options=["No", "Yes"], index=0)
        deck = st.selectbox("Deck Letter", options=["U (Unknown)", "A", "B", "C", "D", "E", "F", "G"], index=0)
        deck_code = deck.split(" ")[0]

    if st.button("Predict Survival Probability 🚀", use_container_width=True):
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        small_fam = 1 if 2 <= family_size <= 4 else 0
        large_fam = 1 if family_size > 4 else 0
        fare_log = np.log1p(fare)
        
        # Build single-sample dictionary matching feature names
        sample_dict = {f: 0 for f in feature_names}
        
        if 'Pclass' in sample_dict: sample_dict['Pclass'] = pclass
        if 'Age' in sample_dict: sample_dict['Age'] = age
        if 'SibSp' in sample_dict: sample_dict['SibSp'] = sibsp
        if 'Parch' in sample_dict: sample_dict['Parch'] = parch
        if 'Fare' in sample_dict: sample_dict['Fare'] = fare
        if 'FamilySize' in sample_dict: sample_dict['FamilySize'] = family_size
        if 'IsAlone' in sample_dict: sample_dict['IsAlone'] = is_alone
        if 'SmallFamily' in sample_dict: sample_dict['SmallFamily'] = small_fam
        if 'LargeFamily' in sample_dict: sample_dict['LargeFamily'] = large_fam
        if 'HasCabin' in sample_dict: sample_dict['HasCabin'] = 1 if has_cabin == "Yes" else 0
        if 'Fare_Log' in sample_dict: sample_dict['Fare_Log'] = fare_log
        
        # Encoded features
        if f'Sex_{sex}' in sample_dict: sample_dict[f'Sex_{sex}'] = 1
        if f'Embarked_{embarked_code}' in sample_dict: sample_dict[f'Embarked_{embarked_code}'] = 1
        if f'Title_{title}' in sample_dict: sample_dict[f'Title_{title}'] = 1
        if f'Deck_{deck_code}' in sample_dict: sample_dict[f'Deck_{deck_code}'] = 1
        
        input_df = pd.DataFrame([sample_dict])[feature_names]
        
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if pred == 1:
                st.success("### Status: SURVIVED 🎉")
            else:
                st.error("### Status: PERISHED 💔")
            st.metric(label="Estimated Survival Probability", value=f"{prob:.2%}")
            
        with res_col2:
            st.progress(float(prob))
            st.info(f"**Analysis**: Passenger class **{pclass}** ({sex}, age {age}) has a predicted survival likelihood of **{prob:.1%}** based on our 5-model Soft Voting Classifier Ensemble.")

with tabs[1]:
    st.subheader("Model Performance Figures")
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        if os.path.exists("reports/figures/cv_model_comparison.png"):
            st.image("reports/figures/cv_model_comparison.png", caption="5-Fold Stratified CV Benchmark")
        if os.path.exists("reports/figures/confusion_matrix.png"):
            st.image("reports/figures/confusion_matrix.png", caption="Model Confusion Matrix")
            
    with fig_col2:
        if os.path.exists("reports/figures/feature_importance.png"):
            st.image("reports/figures/feature_importance.png", caption="Top Engineered Feature Importances")
        if os.path.exists("reports/figures/survival_by_sex_pclass.png"):
            st.image("reports/figures/survival_by_sex_pclass.png", caption="Exploratory Analysis: Survival by Class & Gender")

with tabs[2]:
    st.subheader("Pipeline Technical Overview")
    st.json({
        "Model Type": "Soft Voting Ensemble (Random Forest + Gradient Boosting + Extra Trees + SVM + Logistic Regression)",
        "Mean CV Accuracy": "84.17%",
        "Dataset Folds": "5-Fold Stratified K-Fold",
        "Key Features Engineered": ["Title", "FamilySize", "IsAlone", "SmallFamily", "LargeFamily", "HasCabin", "Deck", "Fare_Log"]
    })
