import os
import base64
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from features import build_passenger_record
from reports import available_figures, format_metric_rows, load_cv_results, load_metrics
from data_prep import load_data
from manifest import (
    prepare_manifest, score_manifest, summarise_predictions, validate_manifest
)

st.set_page_config(
    page_title="RMS Titanic — Survival Command & Telemetry Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to convert background image to base64
def get_base64_of_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

hero_bg_b64 = get_base64_of_file("assets/titanic_hero.png")

# Faded Titanic Background CSS
bg_style = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Inter:wght@300;400;600;700&display=swap');

    .stApp {{
        background: 
            linear-gradient(180deg, rgba(5, 12, 24, 0.88) 0%, rgba(3, 8, 16, 0.96) 100%),
            url("data:image/png;base64,{hero_bg_b64}") no-repeat center top fixed;
        background-size: cover;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Hero Title Card */
    .hero-card {{
        background: rgba(11, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.7);
    }}
    .hero-title {{
        font-family: 'Cinzel', serif;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #FFFFFF 0%, #D4AF37 50%, #00E5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        margin: 0;
    }}
    .hero-subtitle {{
        font-family: 'Cinzel', serif;
        font-size: 1.05rem;
        color: #94a3b8;
        letter-spacing: 2px;
        margin-top: 0.4rem;
    }}
    .gold-badge {{
        display: inline-block;
        background: rgba(212, 175, 55, 0.15);
        border: 1px solid #D4AF37;
        color: #F1C40F;
        padding: 0.2rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }}

    /* Manifest Ticket Card */
    .ticket-card {{
        background: rgba(11, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
    }}
    .ticket-header {{
        font-family: 'Cinzel', serif;
        font-size: 1.3rem;
        color: #D4AF37;
        letter-spacing: 1.5px;
        border-bottom: 1px dashed rgba(212, 175, 55, 0.3);
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
    }}
    
    /* Result Displays */
    .result-survived {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 78, 59, 0.5) 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.25);
    }}
    .result-perished {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(127, 29, 29, 0.5) 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.25);
    }}
    .result-title {{
        font-family: 'Cinzel', serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }}

    /* Metric Badges */
    .metric-badge {{
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 12px;
        padding: 0.85rem;
        text-align: center;
    }}
    .metric-val {{
        font-size: 1.6rem;
        font-weight: 800;
        color: #00E5FF;
    }}
    .metric-lbl {{
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
</style>
"""

st.markdown(bg_style, unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        return None, None
    data = joblib.load(model_path)
    return data['model'], data['feature_names']

model, feature_names = load_pipeline()

# Hero Header Card
st.markdown("""
<div class="hero-card">
    <div class="gold-badge">⚓ White Star Line • RMS Titanic Command Center</div>
    <h1 class="hero-title">Titanic Survival Simulator</h1>
    <div class="hero-subtitle">1912 Evacuation Risk Analytics & AI Telemetry Engine</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Trained ML model artifact not found. Please run `python src/train.py` first!")
    st.stop()

tabs = st.tabs([
    "🎫 Passenger Manifest Simulator",
    "🚢 Batch Manifest Scoring",
    "🧊 Oceanic Telemetry & ML Benchmark",
    "📜 Technical Spec & Logs",
])

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
        cabin = st.text_input("🔑 Recorded Cabin (blank if none)", value="", placeholder="e.g. C85")
        cabin = cabin.strip() or None

    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚨 SIMULATE RESCUE SURVIVAL PROBABILITY", width="stretch"):
        input_df, engineered = build_passenger_record(
            feature_names,
            pclass=pclass,
            sex=sex,
            age=float(age),
            title=title,
            sibsp=int(sibsp),
            parch=int(parch),
            fare=float(fare),
            embarked=embarked_code,
            cabin=cabin,
        )
        
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        family_size = int(engineered["FamilySize"].iloc[0])
        deck = str(engineered["Deck"].iloc[0])
        age_group = str(engineered["AgeGroup"].iloc[0])
        has_cabin = int(engineered["HasCabin"].iloc[0])
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1.2, 2])
        
        pclass_ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(pclass, f"{pclass}th")
        
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
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{pclass_ordinal} Class</div><div class="metric-lbl">Deck Level Access</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{sex.title()}</div><div class="metric-lbl">Evacuation Priority</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{family_size}</div><div class="metric-lbl">Family Unit Size</div></div>', unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(prob))
            
            st.caption(
                f"Encoded context: AgeGroup '{age_group}', Deck '{deck}', "
                f"HasCabin {has_cabin}, Title '{title}', Embarked '{embarked_code}'."
            )
            
            if sex == "female":
                st.info("💡 **Evacuation Protocol**: Maritime 'Women & Children First' protocol significantly boosted survival probability for female passengers.")
            elif pclass == 3:
                st.warning("⚠️ **Steerage Warning**: 3rd Class passengers faced severe delays reaching upper boat decks through flooded watertight compartments.")
            elif title == "Master":
                st.success("👶 **Child Priority**: Young boys ('Master') received preferential evacuation access.")

with tabs[1]:
    st.markdown("### 🚢 Batch Manifest Scoring")
    st.caption(
        "Score a whole passenger list through the same encoder the model was trained on, "
        "rather than one passenger at a time."
    )
    st.markdown(
        '<div class="ticket-card">'
        '<div class="ticket-header">📥 Supply a Passenger Manifest</div>',
        unsafe_allow_html=True,
    )
    
    source = st.radio(
        "Manifest source",
        options=["Kaggle test set (bundled)", "Upload my own CSV"],
        horizontal=True,
    )
    
    uploaded = None
    if source == "Upload my own CSV":
        uploaded = st.file_uploader("Passenger CSV", type=["csv"], width="stretch")
        st.caption(
            "Required columns: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`. "
            "Optional: `PassengerId`, `Title` (one of Mr/Mrs/Miss/Master/Rare), `Cabin`. "
            "A `Title` column is accepted in place of `Name`. Missing `Age` and `Fare` are "
            "imputed from the training set."
        )
    else:
        st.caption(
            "Scores the 418 bundled test passengers. This is the same set that "
            "`submission.csv` is generated from."
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    manifest_df = None
    if source == "Upload my own CSV":
        if uploaded is not None:
            try:
                manifest_df = pd.read_csv(uploaded)
            except Exception as exc:  # noqa: BLE001 - surface any parser failure to the user
                st.error(f"Could not parse that file as CSV: {exc}")
    else:
        test_path = os.path.join("test.csv") if os.path.exists("test.csv") else os.path.join("data", "raw", "test.csv")
        if os.path.exists(test_path):
            manifest_df = pd.read_csv(test_path)
        else:
            st.warning("Bundled test set not found on disk.")
    
    if manifest_df is not None:
        problems = validate_manifest(manifest_df)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            st.success(f"Manifest accepted: {len(manifest_df)} passenger rows.")
            if st.button("⚓ SCORE ENTIRE MANIFEST", width="stretch"):
                with st.spinner("Encoding manifest and scoring..."):
                    train_df, _ = load_data()
                    aligned, engineered = prepare_manifest(manifest_df, feature_names, train_df)
                    probs, preds = score_manifest(model, aligned)
                    ids = (
                        engineered["PassengerId"]
                        if "PassengerId" in engineered.columns
                        else pd.RangeIndex(1, len(engineered) + 1)
                    )
                    summary = summarise_predictions(probs, preds, ids)
                
                st.session_state["manifest_summary"] = summary
                rescued = int(summary["Survived"].sum())
                rate = float(summary["SurvivalProbability"].mean())
                card_cols = st.columns(4)
                with card_cols[0]:
                    st.markdown(
                        f'<div class="metric-badge"><div class="metric-val">{len(summary)}</div>'
                        '<div class="metric-lbl">Passengers Scored</div></div>',
                        unsafe_allow_html=True,
                    )
                with card_cols[1]:
                    st.markdown(
                        f'<div class="metric-badge"><div class="metric-val">{rescued}</div>'
                        '<div class="metric-lbl">Rescued</div></div>',
                        unsafe_allow_html=True,
                    )
                with card_cols[2]:
                    st.markdown(
                        f'<div class="metric-badge"><div class="metric-val">{len(summary) - rescued}</div>'
                        '<div class="metric-lbl">Perished</div></div>',
                        unsafe_allow_html=True,
                    )
                with card_cols[3]:
                    st.markdown(
                        f'<div class="metric-badge"><div class="metric-val">{rate:.1%}</div>'
                        '<div class="metric-lbl">Mean Survival Odds</div></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("<br>", unsafe_allow_html=True)
                
                view = summary.copy()
                view["SurvivalProbability"] = view["SurvivalProbability"].map("{:.2%}".format)
                st.dataframe(
                    view,
                    width="stretch",
                    hide_index=True,
                    height=420,
                    column_config={
                        "PassengerId": st.column_config.NumberColumn("Passenger ID"),
                        "SurvivalProbability": st.column_config.TextColumn("Survival Odds"),
                        "Survived": st.column_config.NumberColumn("Survived (0/1)"),
                        "Outcome": st.column_config.TextColumn("Outcome"),
                    },
                )
                
                kaggle_cols = ["PassengerId", "Survived"]
                st.download_button(
                    "⬇️ Download Kaggle-format CSV",
                    summary[kaggle_cols].to_csv(index=False).encode("utf-8"),
                    file_name="manifest_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )
                st.download_button(
                    "⬇️ Download full table with probabilities",
                    summary.to_csv(index=False).encode("utf-8"),
                    file_name="manifest_predictions_detailed.csv",
                    mime="text/csv",
                )

with tabs[2]:
    st.markdown("### 🧊 Oceanic Model Performance & Analytics")
    st.caption("Live figures and numbers read straight from the artefacts written by the last pipeline run.")
    
    metrics = load_metrics()
    cv_results = load_cv_results()
    figures = available_figures()
    
    if metrics is None and cv_results is None:
        st.info(
            "📉 No benchmark artefacts found yet. Run `python src/train.py` to generate "
            "`reports/metrics.json`, `reports/cv_results.csv` and the figures below."
        )
    else:
        if metrics:
            metric_rows = format_metric_rows(metrics.get("oof_metrics", {}))
            if metric_rows:
                resub = metrics.get("resubstitution_metrics", {}).get("Accuracy")
                st.markdown("#### Out-of-Fold Performance")
                st.caption(
                    "Measured with `cross_val_predict`, so every prediction comes from a model "
                    "that never saw that row."
                    + (f" Resubstitution accuracy is {resub:.2%}, which is not a "
                       "generalisation estimate." if isinstance(resub, float) else "")
                )
                card_cols = st.columns(len(metric_rows))
                for col, (label, value) in zip(card_cols, metric_rows):
                    with col:
                        st.markdown(
                            f'<div class="metric-badge"><div class="metric-val">{value:.2%}</div>'
                            f'<div class="metric-lbl">{label}</div></div>',
                            unsafe_allow_html=True,
                        )
                st.markdown("<br>", unsafe_allow_html=True)
        
        if cv_results is not None:
            st.markdown("#### Model Benchmark")
            st.caption(
                "5-Fold Stratified CV across all 891 training rows. Per-fold scores are in "
                "`reports/cv_results.csv`."
            )
            display_df = cv_results.copy()
            for col in ["Mean_CV_Accuracy", "Std_Dev"] + [
                c for c in display_df.columns if c.startswith("Fold")
            ]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map("{:.2%}".format)
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Model": st.column_config.TextColumn("Model", width="medium"),
                    "Mean_CV_Accuracy": st.column_config.TextColumn("Mean CV Accuracy"),
                    "Std_Dev": st.column_config.TextColumn("Std Dev"),
                },
            )
        
        st.markdown("#### Generated Figures")
        figure_captions = {
            "cv_model_comparison": "5-Fold Stratified Cross-Validation Benchmark",
            "roc_pr_curves": "ROC and Precision-Recall Curves (out-of-fold)",
            "threshold_sweep": "Decision Threshold Sweep",
            "feature_importance": "Top Feature Importances (tree ensemble)",
            "confusion_matrix": "Confusion Matrix (out-of-fold)",
        }
        plot_order = [key for key in figure_captions if key in figures]
        if not plot_order:
            st.caption("No figures found. Run `python src/train.py` to generate them.")
        for chunk_start in range(0, len(plot_order), 2):
            chunk = plot_order[chunk_start:chunk_start + 2]
            for col, key in zip(st.columns(len(chunk)), chunk):
                with col:
                    st.image(figures[key], caption=figure_captions[key])
        
        if os.path.exists("assets/iceberg.png"):
            st.image("assets/iceberg.png", caption="North Atlantic Oceanic Environment", width=260)

with tabs[3]:
    st.markdown("### 📜 System Architecture & Technical Specifications")
    spec = {
        "System Name": "RMS Titanic Survival Prediction Engine",
        "Primary Model": "Voting Classifier Ensemble",
        "Ensemble Sub-Models": ["Random Forest", "Tuned Gradient Boosting", "Extra Trees", "Support Vector Classifier", "Logistic Regression"],
        "Dataset Folds": "5-Fold Stratified K-Fold",
        "Engineered Features": ["Title Extraction", "FamilySize", "IsAlone", "SmallFamily", "LargeFamily", "HasCabin", "Deck Level", "Fare Log-Transform"],
        "Target Variable": "Survived (0 = Perished, 1 = Rescued)",
    }
    if metrics:
        oof = metrics.get("oof_metrics", {})
        spec["Selected Model"] = metrics.get("selected_model", spec["Primary Model"])
        spec["Cross-Validation Benchmark"] = f"{metrics.get('cv_mean_accuracy', 0):.2%} mean accuracy"
        spec["Cross-Validation Std Dev"] = f"+/- {metrics.get('cv_std_accuracy', 0):.2%}"
        if "ROC-AUC" in oof:
            spec["Out-of-Fold ROC-AUC"] = f"{oof['ROC-AUC']:.4f}"
        spec["Training Rows"] = metrics.get("n_train")
        spec["Test Rows"] = metrics.get("n_test")
    else:
        spec["Cross-Validation Benchmark"] = "Run `python src/train.py` to populate"
    st.json(spec)
