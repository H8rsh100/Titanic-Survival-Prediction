import os
import base64
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="RMS Titanic — Survival Command & Telemetry Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to encode local image to base64 for CSS background
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Inject Atmospheric CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 20%, #0e2338 0%, #071322 50%, #030810 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Banner Styling */
    .hero-container {
        position: relative;
        background: linear-gradient(180deg, rgba(7, 19, 34, 0.4) 0%, rgba(3, 8, 16, 0.95) 100%);
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7), inset 0 0 30px rgba(0, 229, 255, 0.05);
        overflow: hidden;
    }
    .hero-title {
        font-family: 'Cinzel', serif;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #FFF 0%, #D4AF37 50%, #00E5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        margin: 0;
        text-shadow: 0 10px 30px rgba(0, 229, 255, 0.2);
    }
    .hero-subtitle {
        font-family: 'Cinzel', serif;
        font-size: 1.15rem;
        color: #94a3b8;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    .gold-badge {
        display: inline-block;
        background: rgba(212, 175, 55, 0.15);
        border: 1px solid #D4AF37;
        color: #F1C40F;
        padding: 0.25rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Ticket Container */
    .ticket-card {
        background: rgba(13, 27, 46, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
    }
    .ticket-header {
        font-family: 'Cinzel', serif;
        font-size: 1.4rem;
        color: #D4AF37;
        letter-spacing: 1.5px;
        border-bottom: 1px dashed rgba(212, 175, 55, 0.3);
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    /* Result Cards */
    .result-survived {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 78, 59, 0.4) 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }
    .result-perished {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(127, 29, 29, 0.4) 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
    }
    .result-title {
        font-family: 'Cinzel', serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    /* Metric Badges */
    .metric-badge {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00E5FF;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        return None, None
    data = joblib.load(model_path)
    return data['model'], data['feature_names']

model, feature_names = load_pipeline()

# Hero Banner Section
st.markdown("""
<div class="hero-container">
    <div class="gold-badge">⚓ White Star Line • RMS Titanic Command Center</div>
    <h1 class="hero-title">Titanic Survival Simulator</h1>
    <div class="hero-subtitle">1912 Evacuation Risk Analytics & AI Telemetry Engine</div>
</div>
""", unsafe_allow_html=True)

if os.path.exists("assets/titanic_hero.png"):
    st.image("assets/titanic_hero.png", use_container_width=True)

if model is None:
    st.error("⚠️ Trained ML model artifact not found. Please run `python src/train.py` first!")
    st.stop()

tabs = st.tabs(["🎫 Passenger Manifest Simulator", "🧊 Oceanic Telemetry & ML Benchmark", "📜 Technical Spec & Logs"])

with tabs[0]:
    st.markdown('<div class="ticket-card">', unsafe_allow_html=True)
    st.markdown('<div class="ticket-header">📋 Boarding Ticket Manifest Details</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pclass = st.selectbox("🎟️ Ticket Class (Pclass)", options=[1, 2, 3], index=2, help="1st Class (Upper Deck), 2nd Class (Middle), 3rd Class (Lower Steerage)")
        sex = st.selectbox("👤 Passenger Sex", options=["female", "male"], index=1)
        age = st.slider("🎂 Passenger Age", min_value=0, max_value=80, value=28)
        title = st.selectbox("🏷️ Passenger Title", options=["Mr", "Mrs", "Miss", "Master", "Rare"], index=0)

    with col2:
        sibsp = st.number_input("👨‍👩‍👧 Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
        parch = st.number_input("👶 Parents / Children Aboard", min_value=0, max_value=6, value=0)
        fare = st.slider("💰 Ticket Fare (£ Sterling 1912)", min_value=0.0, max_value=500.0, value=32.2)

    with col3:
        embarked = st.selectbox("⚓ Port of Embarkation", options=["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"], index=0)
        embarked_code = embarked.split(" ")[0]
        has_cabin = st.selectbox("🔑 Recorded Cabin Assigned?", options=["No", "Yes"], index=0)
        deck = st.selectbox("⛵ Cabin Deck Level", options=["U (Unknown/Steerage)", "A", "B", "C", "D", "E", "F", "G"], index=0)
        deck_code = deck.split(" ")[0]

    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚨 SIMULATE RESCUE SURVIVAL PROBABILITY", use_container_width=True):
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        small_fam = 1 if 2 <= family_size <= 4 else 0
        large_fam = 1 if family_size > 4 else 0
        fare_log = np.log1p(fare)
        
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
        
        if f'Sex_{sex}' in sample_dict: sample_dict[f'Sex_{sex}'] = 1
        if f'Embarked_{embarked_code}' in sample_dict: sample_dict[f'Embarked_{embarked_code}'] = 1
        if f'Title_{title}' in sample_dict: sample_dict[f'Title_{title}'] = 1
        if f'Deck_{deck_code}' in sample_dict: sample_dict[f'Deck_{deck_code}'] = 1
        
        input_df = pd.DataFrame([sample_dict])[feature_names]
        
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1.2, 2])
        
        with res_col1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-survived">
                    <div class="result-title" style="color: #10B981;">RESCUED / SURVIVED 🎉</div>
                    <p style="margin-top: 0.5rem; color: #a7f3d0;">Allocated Lifeboat Space</p>
                    <h2 style="color: #FFF; margin: 0.5rem 0;">{prob:.1%}</h2>
                    <span style="font-size: 0.8rem; color: #6ee7b7;">Calculated Survival Likelihood</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-perished">
                    <div class="result-title" style="color: #EF4444;">PERISHED IN DISASTER 💔</div>
                    <p style="margin-top: 0.5rem; color: #fca5a5;">Critical Lifeboat Deficit</p>
                    <h2 style="color: #FFF; margin: 0.5rem 0;">{prob:.1%}</h2>
                    <span style="font-size: 0.8rem; color: #f87171;">Calculated Survival Likelihood</span>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("### 📊 Key Telemetry Factors & Decision Drivers")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{pclass}st Class</div><div class="metric-lbl">Deck Level Access</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{sex.title()}</div><div class="metric-lbl">Evacuation Priority</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{family_size}</div><div class="metric-lbl">Family Unit Size</div></div>', unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(prob))
            
            if sex == "female":
                st.info("💡 **Evacuation Protocol**: Maritime 'Women & Children First' protocol significantly boosted survival probability for female passengers.")
            elif pclass == 3:
                st.warning("⚠️ **Steerage Warning**: 3rd Class passengers faced severe delays reaching upper boat decks through flooded watertight compartments.")
            elif title == "Master":
                st.success("👶 **Child Priority**: Young boys ('Master') received preferential evacuation access.")

with tabs[1]:
    st.markdown("### 🧊 Oceanic Model Performance & Analytics")
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("assets/iceberg.png"):
            st.image("assets/iceberg.png", caption="North Atlantic Oceanic Environment")
        if os.path.exists("reports/figures/cv_model_comparison.png"):
            st.image("reports/figures/cv_model_comparison.png", caption="5-Fold Stratified Cross-Validation Benchmark")
            
    with col_img2:
        if os.path.exists("reports/figures/feature_importance.png"):
            st.image("reports/figures/feature_importance.png", caption="Top Feature Importances (Tree Ensemble)")
        if os.path.exists("reports/figures/confusion_matrix.png"):
            st.image("reports/figures/confusion_matrix.png", caption="Confusion Matrix — Soft Voting Ensemble")

with tabs[2]:
    st.markdown("### 📜 System Architecture & Technical Specifications")
    st.json({
        "System Name": "RMS Titanic Survival Prediction Engine",
        "Primary Model": "Soft Voting Classifier Ensemble",
        "Ensemble Sub-Models": ["Random Forest", "Tuned Gradient Boosting", "Extra Trees", "Support Vector Classifier", "Logistic Regression"],
        "Cross-Validation Benchmark": "84.17% Mean Stratified CV Accuracy",
        "Dataset Folds": "5-Fold Stratified K-Fold",
        "Engineered Features": ["Title Extraction", "FamilySize", "IsAlone", "SmallFamily", "LargeFamily", "HasCabin", "Deck Level", "Fare Log-Transform"],
        "Target Variable": "Survived (0 = Perished, 1 = Rescued)"
    })
