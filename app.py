"""
app.py — Streamlit GUI for Sepsis Prediction
---------------------------------------------
Run: streamlit run app.py

Features:
  • Manual single-patient input (key vitals)
  • Upload a PSV patient file
  • Shows prediction, probability, risk level
  • Displays a risk gauge + time-series plot of vitals
  • Feature importance chart
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))
from inference import SepsisPredictor

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sepsis Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 10px 0;
    }
    .risk-low    { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high   { background-color: #f8d7da; color: #721c24; }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 20px;
        margin: 6px 0;
        border-left: 4px solid #1a73e8;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load predictor (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_predictor(model_name: str):
    return SepsisPredictor(model_name=model_name)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=80)
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["random_forest", "logistic", "gradient_boosting"],
    index=0,
)

input_mode = st.sidebar.radio(
    "Input Mode",
    ["🖊️ Manual Entry", "📁 Upload PSV File"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Evaluation Metrics**")
metrics_path = os.path.join("outputs", "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    st.sidebar.metric("Accuracy",  f"{m.get('accuracy',0):.3f}")
    st.sidebar.metric("Recall",    f"{m.get('recall',0):.3f}")
    st.sidebar.metric("F1-Score",  f"{m.get('f1',0):.3f}")
    st.sidebar.metric("ROC-AUC",   f"{m.get('roc_auc',0):.3f}")
else:
    st.sidebar.info("Run train.py first to see model metrics.")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🏥 Sepsis Early Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Healthcare AI Challenge — ICU Sepsis Risk Assessment</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Load predictor
# ─────────────────────────────────────────────────────────────────────────────

try:
    predictor = load_predictor(model_choice)
    model_ready = True
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    model_ready = False
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build patient DataFrame from manual inputs
# ─────────────────────────────────────────────────────────────────────────────

VITAL_DEFAULTS = {
    "HR":     80,   "O2Sat": 98,  "Temp":  37.0,
    "SBP":   120,   "MAP":   80,  "DBP":   70,
    "Resp":   18,   "Glucose": 100,
}

LAB_DEFAULTS = {
    "pH":         7.40, "PaCO2":    40.0, "HCO3":      24.0,
    "Lactate":     1.0, "WBC":      10.0, "Creatinine":  1.0,
    "Hgb":        12.0, "Platelets": 250, "Potassium":   4.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Manual input mode
# ─────────────────────────────────────────────────────────────────────────────

patient_df = None

if input_mode == "🖊️ Manual Entry":
    st.subheader("Patient Vital Signs & Lab Values")
    st.caption("Enter values for a single time step (or multiple rows by adding hours below).")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🫀 Vitals**")
        HR     = st.number_input("Heart Rate (bpm)",    value=80.0, min_value=0.0, max_value=300.0)
        O2Sat  = st.number_input("O₂ Saturation (%)",  value=98.0, min_value=0.0, max_value=100.0)
        Temp   = st.number_input("Temperature (°C)",   value=37.0, min_value=30.0, max_value=45.0)
        Resp   = st.number_input("Respiration Rate",   value=18.0, min_value=0.0, max_value=60.0)

    with col2:
        st.markdown("**💉 Blood Pressure**")
        SBP    = st.number_input("SBP (mmHg)",  value=120.0, min_value=0.0, max_value=300.0)
        DBP    = st.number_input("DBP (mmHg)",  value=70.0,  min_value=0.0, max_value=200.0)
        MAP    = st.number_input("MAP (mmHg)",  value=80.0,  min_value=0.0, max_value=250.0)

    with col3:
        st.markdown("**🧪 Key Labs**")
        pH         = st.number_input("pH",            value=7.40, min_value=6.0, max_value=8.0, step=0.01)
        Lactate    = st.number_input("Lactate",        value=1.0,  min_value=0.0, max_value=30.0)
        WBC        = st.number_input("WBC (k/μL)",     value=10.0, min_value=0.0, max_value=100.0)
        Creatinine = st.number_input("Creatinine",     value=1.0,  min_value=0.0, max_value=30.0)
        Glucose    = st.number_input("Glucose (mg/dL)",value=100.0,min_value=0.0, max_value=1000.0)

    st.markdown("**👤 Demographics**")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        Age    = st.number_input("Age", value=65, min_value=0, max_value=120)
    with dcol2:
        Gender = st.selectbox("Gender", ["Male (1)", "Female (0)"])
        Gender = 1 if "Male" in Gender else 0
    with dcol3:
        ICULOS = st.number_input("Hours in ICU (ICULOS)", value=6, min_value=1, max_value=336)

    # Build single-row patient DataFrame
    patient_df = pd.DataFrame([{
        "HR": HR, "O2Sat": O2Sat, "Temp": Temp, "SBP": SBP, "MAP": MAP,
        "DBP": DBP, "Resp": Resp, "EtCO2": np.nan, "BaseExcess": np.nan,
        "HCO3": np.nan, "FiO2": np.nan, "pH": pH, "PaCO2": np.nan,
        "SaO2": np.nan, "AST": np.nan, "BUN": np.nan, "Alkalinephos": np.nan,
        "Calcium": np.nan, "Chloride": np.nan, "Creatinine": Creatinine,
        "Bilirubin_direct": np.nan, "Glucose": Glucose, "Lactate": Lactate,
        "Magnesium": np.nan, "Phosphate": np.nan, "Potassium": np.nan,
        "Bilirubin_total": np.nan, "TroponinI": np.nan, "Hct": np.nan,
        "Hgb": np.nan, "PTT": np.nan, "WBC": WBC, "Fibrinogen": np.nan,
        "Platelets": np.nan, "Age": Age, "Gender": Gender,
        "Unit1": 1, "Unit2": 0, "HospAdmTime": -6.0, "ICULOS": ICULOS,
        "SepsisLabel": 0,
    }])

else:
    # ── Upload mode ────────────────────────────────────────────────────────────
    st.subheader("Upload Patient PSV File")
    uploaded = st.file_uploader("Upload a .psv patient file", type=["psv", "csv", "txt"])
    if uploaded:
        sep = "|" if uploaded.name.endswith(".psv") else ","
        patient_df = pd.read_csv(uploaded, sep=sep, na_values=["NaN", "nan", ""])
        st.success(f"Loaded {len(patient_df)} time-steps for this patient.")

        # Show raw data
        with st.expander("📋 View raw uploaded data"):
            st.dataframe(patient_df.head(20))

        # Vitals plot
        vital_cols = [c for c in ["HR","O2Sat","Temp","Resp","SBP","MAP"] if c in patient_df.columns]
        if vital_cols and "ICULOS" in patient_df.columns:
            fig = px.line(
                patient_df, x="ICULOS", y=vital_cols,
                title="Vital Signs Over Time",
                labels={"value": "Value", "ICULOS": "Hours in ICU"},
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction button
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
predict_btn = st.button("🔍 Predict Sepsis Risk", type="primary", use_container_width=True)

if predict_btn and patient_df is not None:
    with st.spinner("Running inference..."):
        t0     = time.perf_counter()
        result = predictor.predict(patient_df)
        inf_ms = (time.perf_counter() - t0) * 1000

    pred  = result["prediction"]
    prob  = result["probability"]
    risk  = result["risk_level"]
    conf  = result["confidence_pct"]

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Prediction", "⚠️ SEPSIS" if pred == 1 else "✅ No Sepsis")
    with col_b:
        st.metric("Probability", f"{conf:.1f}%")
    with col_c:
        st.metric("Inference Time", f"{inf_ms:.1f} ms")

    # Risk card
    risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[risk]
    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk]
    st.markdown(
        f'<div class="risk-card {risk_class}">{risk_emoji} Risk Level: {risk}</div>',
        unsafe_allow_html=True,
    )

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={"text": "Sepsis Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#1a73e8"},
            "steps": [
                {"range": [0,  35], "color": "#d4edda"},
                {"range": [35, 65], "color": "#fff3cd"},
                {"range": [65, 100],"color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 65,
            },
        },
        delta={"reference": 50},
        number={"suffix": "%", "valueformat": ".1f"},
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Clinical guidance
    st.markdown("**📋 Clinical Guidance**")
    if risk == "Low":
        st.success("Low sepsis probability. Continue standard monitoring.")
    elif risk == "Medium":
        st.warning("Moderate sepsis risk. Consider enhanced monitoring and early cultures.")
    else:
        st.error("High sepsis risk. Immediate clinical evaluation strongly recommended.")

    st.caption("⚠️ This is an AI-assisted decision support tool only. Always defer to clinical judgment.")

elif predict_btn and patient_df is None:
    st.error("Please provide patient data first (manual entry or file upload).")

# ─────────────────────────────────────────────────────────────────────────────
# Feature importance panel
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📊 Model Insights")

imp_path = os.path.join("outputs", "feature_importances.csv")
if os.path.exists(imp_path):
    imp_df = pd.read_csv(imp_path).head(20)
    fig_imp = px.bar(
        imp_df[::-1], x="importance", y="feature",
        orientation="h",
        title="Top 20 Feature Importances",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Feature importances not available yet. Run train.py to generate them.")

# ROC curve
roc_path = os.path.join("outputs", f"roc_curve_{model_choice}.png")
if os.path.exists(roc_path):
    col_roc, col_cm = st.columns(2)
    with col_roc:
        st.image(roc_path, caption="ROC Curve", width=700)
    cm_path = os.path.join("outputs", f"confusion_matrix_{model_choice}.png")
    if os.path.exists(cm_path):
        with col_cm:
            st.image(cm_path, caption="Confusion Matrix", width=700)
