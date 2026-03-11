import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load model & scaler
# -----------------------------
model = pickle.load(open("KNN_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# -----------------------------
# Professional Medical CSS Theme
# -----------------------------
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.main {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
}

body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
}

h1 {
    color: #1a3a52;
    text-align: center;
    font-weight: 800;
    font-size: 2.5em;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}

h2 {
    color: #1a3a52;
    font-weight: 700;
    margin-top: 15px;
}

.subtitle {
    text-align: center;
    color: #4a6b7c;
    margin-bottom: 30px;
    font-size: 1.1em;
    font-weight: 500;
    letter-spacing: 0.3px;
}

.card {
    background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
    padding: 25px 28px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    margin-bottom: 20px;
    border: 1px solid rgba(26, 58, 82, 0.05);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    border-color: rgba(26, 58, 82, 0.1);
}

.section-title {
    color: #1a3a52;
    font-weight: 700;
    margin-bottom: 15px;
    font-size: 1.15em;
    display: flex;
    align-items: center;
    gap: 10px;
}

.result-ok {
    background: linear-gradient(135deg, #d4f1e4 0%, #e8f7f1 100%);
    padding: 24px;
    border-radius: 14px;
    color: #0d5f47;
    font-weight: 700;
    text-align: center;
    border: 2px solid #a7e5d4;
    font-size: 1.1em;
    margin-top: 20px;
    box-shadow: 0 6px 20px rgba(13, 95, 71, 0.15);
}

.result-bad {
    background: linear-gradient(135deg, #fde4e4 0%, #fdecec 100%);
    padding: 24px;
    border-radius: 14px;
    color: #7a2c2c;
    font-weight: 700;
    text-align: center;
    border: 2px solid #f5c2c7;
    font-size: 1.1em;
    margin-top: 20px;
    box-shadow: 0 6px 20px rgba(122, 44, 44, 0.15);
}

.result-recommend {
    background: rgba(255, 193, 7, 0.1);
    padding: 16px;
    border-radius: 10px;
    color: #826d03;
    font-size: 0.95em;
    margin-top: 12px;
    border-left: 4px solid #ffc107;
}

.input-label {
    color: #1a3a52;
    font-weight: 600;
    margin-bottom: 8px;
    display: block;
}

.metric-description {
    color: #5a7a8c;
    font-size: 0.9em;
    margin-top: 4px;
}

.footer {
    text-align: center;
    color: #5a7a8c;
    font-size: 14px;
    margin-top: 40px;
    padding: 20px;
    background: rgba(26, 58, 82, 0.03);
    border-radius: 12px;
    letter-spacing: 0.3px;
}

.btn-predict {
    background: linear-gradient(135deg, #1a3a52 0%, #2d5a7a 100%);
    color: white;
    font-weight: 700;
    padding: 12px 24px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    font-size: 1.05em;
    transition: all 0.3s ease;
    width: 100%;
}

.btn-predict:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(26, 58, 82, 0.3);
}

.info-box {
    background: linear-gradient(135deg, rgba(26, 58, 82, 0.05) 0%, rgba(26, 58, 82, 0.02) 100%);
    padding: 16px;
    border-radius: 12px;
    border-left: 4px solid #1a3a52;
    margin-bottom: 20px;
    color: #4a6b7c;
    font-size: 0.95em;
}

.warning-box {
    background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%);
    padding: 16px;
    border-radius: 12px;
    border-left: 4px solid #ff9800;
    margin-bottom: 20px;
    color: #5a4d1a;
    font-size: 0.95em;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(26, 58, 82, 0.1) 50%, transparent 100%);
    margin: 25px 0;
}

.header-container {
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 25%, #e74c3c 50%, #c0392b 75%, #e74c3c 100%);
    padding: 40px 30px;
    border-radius: 20px;
    box-shadow: 0 15px 50px rgba(231, 76, 60, 0.3);
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.15) 0%, transparent 50%);
    pointer-events: none;
}

.header-container h1 {
    color: #ffffff;
    text-align: center;
    font-weight: 900;
    font-size: 3.5em;
    margin: 0;
    text-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    letter-spacing: 1px;
    position: relative;
    z-index: 1;
}

.header-container .subtitle {
    color: rgba(255, 255, 255, 0.95);
    margin-top: 10px;
    font-size: 1.3em;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    position: relative;
    z-index: 1;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a3a52 0%, #2d5a7a 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(26, 58, 82, 0.3) !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class='header-container'>
    <h1>❤️ Heart Disease Prediction System</h1>
    <div class='subtitle'>Advanced ML-Based Clinical Assessment Tool</div>
</div>
""", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class='info-box'>
📋 This tool uses a K-Nearest Neighbors (KNN) machine learning model trained on medical data to assess heart disease risk based on patient health metrics. Always consult with a cardiologist for definitive diagnosis.
</div>
""", unsafe_allow_html=True)

# =========================
# Category mappings
# =========================
sex_map = {0: "Female", 1: "Male"}
cp_map = {
    0: "Typical angina",
    1: "Atypical angina",
    2: "Non-anginal pain",
    3: "Asymptomatic"
}
fbs_map = {0: "≤120 mg/dl", 1: ">120 mg/dl"}
restecg_map = {
    0: "Normal",
    1: "ST-T abnormality",
    2: "Left ventricular hypertrophy"
}
exang_map = {0: "No", 1: "Yes"}
slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
ca_map = {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"}
thal_map = {0: "Unknown", 1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}

# Reverse
sex_rev = {v:k for k,v in sex_map.items()}
cp_rev = {v:k for k,v in cp_map.items()}
fbs_rev = {v:k for k,v in fbs_map.items()}
restecg_rev = {v:k for k,v in restecg_map.items()}
exang_rev = {v:k for k,v in exang_map.items()}
slope_rev = {v:k for k,v in slope_map.items()}
ca_rev = {v:k for k,v in ca_map.items()}
thal_rev = {v:k for k,v in thal_map.items()}

# -----------------------------
# Patient Info Card
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>👤 Patient Demographics</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<span class='input-label'>Age</span>", unsafe_allow_html=True)
    age = st.number_input("Age (years)", 1, 120, 45, label_visibility="collapsed")
    st.markdown("<div class='metric-description'>Patient age in years</div>", unsafe_allow_html=True)
    
with col2:
    st.markdown("<span class='input-label'>Sex</span>", unsafe_allow_html=True)
    sex_text = st.selectbox("Sex", list(sex_map.values()), label_visibility="collapsed", key="sex_select")
    st.markdown("<div class='metric-description'>Biological sex</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<span class='input-label'>Blood Pressure</span>", unsafe_allow_html=True)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, label_visibility="collapsed")
    st.markdown("<div class='metric-description'>mm Hg (resting)</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<span class='input-label'>Cholesterol</span>", unsafe_allow_html=True)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200, label_visibility="collapsed")
    st.markdown("<div class='metric-description'>mg/dl</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Clinical Measures Card
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🩺 Clinical Measurements & Findings</div>", unsafe_allow_html=True)

# Row 1: Primary metrics
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
with col_c1:
    st.markdown("<span class='input-label'>Max Heart Rate</span>", unsafe_allow_html=True)
    thalach = st.number_input("Maximum Heart Rate", 60, 220, 150, label_visibility="collapsed", key="thalach")
    st.markdown("<div class='metric-description'>bpm achieved</div>", unsafe_allow_html=True)

with col_c2:
    st.markdown("<span class='input-label'>ST Depression</span>", unsafe_allow_html=True)
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, label_visibility="collapsed", key="oldpeak")
    st.markdown("<div class='metric-description'>ECG measurement</div>", unsafe_allow_html=True)

with col_c3:
    st.markdown("<span class='input-label'>Chest Pain Type</span>", unsafe_allow_html=True)
    cp_text = st.selectbox("Chest Pain Type", list(cp_map.values()), label_visibility="collapsed", key="cp_select")

with col_c4:
    st.markdown("<span class='input-label'>ST Slope</span>", unsafe_allow_html=True)
    slope_text = st.selectbox("ST Segment Slope", list(slope_map.values()), label_visibility="collapsed", key="slope_select")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Row 2: Secondary metrics
col_c5, col_c6, col_c7, col_c8, col_c9 = st.columns(5)
with col_c5:
    st.markdown("<span class='input-label'>Fasting BS</span>", unsafe_allow_html=True)
    fbs_text = st.selectbox("Fasting Blood Sugar", list(fbs_map.values()), label_visibility="collapsed", key="fbs_select")

with col_c6:
    st.markdown("<span class='input-label'>Resting ECG</span>", unsafe_allow_html=True)
    restecg_text = st.selectbox("Resting ECG", list(restecg_map.values()), label_visibility="collapsed", key="restecg_select")

with col_c7:
    st.markdown("<span class='input-label'>Exercise Angina</span>", unsafe_allow_html=True)
    exang_text = st.selectbox("Exercise Induced Angina", list(exang_map.values()), label_visibility="collapsed", key="exang_select")

with col_c8:
    st.markdown("<span class='input-label'>Thallium Test</span>", unsafe_allow_html=True)
    thal_text = st.selectbox("Thallium Test", list(thal_map.values()), label_visibility="collapsed", key="thal_select")

with col_c9:
    st.markdown("<span class='input-label'>Major Vessels</span>", unsafe_allow_html=True)
    ca_text = st.selectbox("Major Vessels", list(ca_map.values()), label_visibility="collapsed", key="ca_select")

st.markdown("</div>", unsafe_allow_html=True)

# Convert to numeric
sex = sex_rev[sex_text]
cp = cp_rev[cp_text]
fbs = fbs_rev[fbs_text]
restecg = restecg_rev[restecg_text]
exang = exang_rev[exang_text]
slope = slope_rev[slope_text]
ca = ca_rev[ca_text]
thal = thal_rev[thal_text]

# Add spacing before prediction
st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

# Prediction button
col_btn_left, col_btn_middle, col_btn_right = st.columns([1, 2, 1])
with col_btn_middle:
    predict_btn = st.button("🔍 ANALYZE HEART DISEASE RISK", use_container_width=True, key="predict_button")

# Display results
if predict_btn:
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])
    
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    
    # Get prediction probability if available
    try:
        probabilities = model.predict_proba(scaled)[0]
        confidence = max(probabilities) * 100
    except:
        confidence = 85

    if prediction == 1:
        st.markdown(
            "<div class='result-bad'>⚠️ HIGH RISK: Heart Disease Indicators Detected</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='result-recommend'>"
            "<strong>🏥 Recommended Actions:</strong><br>"
            "• Schedule an appointment with a cardiologist immediately<br>"
            "• Undergo detailed cardiac evaluation and ECG testing<br>"
            "• Implement lifestyle modifications (diet, exercise, stress management)<br>"
            "• Monitor blood pressure and cholesterol regularly<br>"
            "• Follow prescribed medications if any"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-ok'>✅ LOW RISK: No Significant Heart Disease Indicators</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='result-recommend'>"
            "<strong>💚 Health Recommendations:</strong><br>"
            "• Maintain a balanced diet rich in fruits and vegetables<br>"
            "• Exercise regularly (at least 30 minutes, 5 days/week)<br>"
            "• Monitor blood pressure and cholesterol annually<br>"
            "• Avoid smoking and limit alcohol consumption<br>"
            "• Continue routine health checkups"
            "</div>",
            unsafe_allow_html=True
        )

# Add divider
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Footer with more information
st.markdown("""
<div class='footer'>
<strong>📖 About This Tool</strong><br>
This application uses a K-Nearest Neighbors (KNN) machine learning model trained on the UCI Heart Disease dataset. 
It analyzes 13 clinical parameters to assess the likelihood of heart disease presence.<br><br>
<strong>⚠️ Disclaimer:</strong> This tool is for educational and informational purposes only and should NOT be used as a substitute for professional medical diagnosis. 
Always consult with a qualified healthcare provider for accurate medical assessment and treatment.<br><br>
<strong>🔒 Privacy:</strong> Your input data is processed locally and is not stored or transmitted.
</div>
""", unsafe_allow_html=True)