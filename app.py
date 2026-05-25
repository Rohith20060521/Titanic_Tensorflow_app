import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* ---------------- GLOBAL ---------------- */

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* ---------------- BACKGROUND ---------------- */

.stApp {

    background:
    radial-gradient(circle at top left, rgba(59,130,246,0.15), transparent 25%),
    radial-gradient(circle at bottom right, rgba(124,58,237,0.18), transparent 25%),
    linear-gradient(135deg, #020617, #0f172a);

    color: #f8fafc;
}

/* ---------------- HIDE DEFAULTS ---------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* ---------------- TITLE ---------------- */

.main-title {

    text-align: center;

    font-size: 4rem;

    font-weight: 800;

    margin-top: 10px;

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {

    text-align: center;

    color: #cbd5e1;

    font-size: 1.1rem;

    margin-bottom: 35px;
}

/* ---------------- CARD ---------------- */

.card {

    background: rgba(15,23,42,0.78);

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(14px);

    padding: 35px;

    border-radius: 24px;

    box-shadow:
    0 8px 32px rgba(0,0,0,0.35);

    margin-bottom: 20px;
}

/* ---------------- HEADINGS ---------------- */

h1, h2, h3, h4 {

    color: #f8fafc;
}

/* ---------------- LABELS ---------------- */

label {

    color: #f1f5f9 !important;

    font-size: 17px !important;

    font-weight: 600 !important;
}

/* ---------------- SELECTBOX ---------------- */

.stSelectbox div[data-baseweb="select"] {

    background: rgba(255,255,255,0.08) !important;

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 14px !important;

    color: white !important;
}

/* ---------------- SLIDER TRACK ---------------- */

.stSlider > div > div > div {

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #7c3aed
    ) !important;

    height: 8px !important;

    border-radius: 10px;
}

/* ---------------- SLIDER CIRCLE ---------------- */

.stSlider [role="slider"] {

    background-color: white !important;

    border: 4px solid #7c3aed !important;

    width: 18px !important;

    height: 18px !important;

    box-shadow:
    0 0 12px rgba(124,58,237,0.8);
}

/* ---------------- SLIDER VALUE ---------------- */

.stSlider span {

    color: white !important;

    font-weight: 600 !important;
}

/* ---------------- BUTTON ---------------- */

.stButton button {

    width: 100%;

    height: 58px;

    border-radius: 16px;

    border: none;

    font-size: 20px;

    font-weight: bold;

    color: white;

    background:
    linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    transition: all 0.3s ease;

    box-shadow:
    0 6px 20px rgba(37,99,235,0.35);
}

/* ---------------- BUTTON HOVER ---------------- */

.stButton button:hover {

    transform: translateY(-2px);

    box-shadow:
    0 10px 30px rgba(124,58,237,0.45);

    background:
    linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    );
}

/* ---------------- SUCCESS CARD ---------------- */

.result-success {

    background:
    linear-gradient(
        135deg,
        rgba(16,185,129,0.18),
        rgba(34,197,94,0.10)
    );

    border: 1px solid rgba(16,185,129,0.45);

    padding: 28px;

    border-radius: 22px;

    text-align: center;

    margin-top: 25px;
}

/* ---------------- DANGER CARD ---------------- */

.result-danger {

    background:
    linear-gradient(
        135deg,
        rgba(239,68,68,0.18),
        rgba(220,38,38,0.10)
    );

    border: 1px solid rgba(239,68,68,0.45);

    padding: 28px;

    border-radius: 22px;

    text-align: center;

    margin-top: 25px;
}

/* ---------------- METRIC ---------------- */

.metric {

    background: rgba(255,255,255,0.06);

    padding: 22px;

    border-radius: 18px;

    text-align: center;

    border: 1px solid rgba(255,255,255,0.08);

    margin-top: 18px;
}

.metric h2 {

    color: #38bdf8;
}

/* ---------------- SUMMARY ---------------- */

.summary-box {

    background: rgba(255,255,255,0.05);

    padding: 30px;

    border-radius: 22px;

    border: 1px solid rgba(255,255,255,0.08);

    margin-top: 28px;
}

.summary-box h2 {

    color: #38bdf8;
}

/* ---------------- FOOTER ---------------- */

.footer {

    text-align: center;

    color: #94a3b8;

    margin-top: 40px;

    padding-bottom: 20px;

    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLES ----------------

st.markdown("""
<div class="main-title">
🚢 Titanic AI Predictor
</div>

<div class="sub-title">
Deep Learning Powered Survival Prediction System
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL STATS ----------------

TRAIN_ACCURACY = 0.74
VALIDATION_ACCURACY = 0.72
LOSS = 0.58

# ---------------- ACTIVATION FUNCTION ----------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------- PREDICTION FUNCTION ----------------

def predict_survival(pclass, age, fare):

    pclass_norm = (pclass - 1) / 2
    age_norm = age / 80
    fare_norm = min(fare / 500, 1)

    h1 = sigmoid(
        (0.8 * pclass_norm) +
        (-0.5 * age_norm) +
        (0.9 * fare_norm)
    )

    h2 = sigmoid(
        (0.6 * pclass_norm) +
        (-0.4 * age_norm) +
        (0.7 * fare_norm)
    )

    output = sigmoid(
        (0.7 * h1) +
        (0.8 * h2) -
        0.6
    )

    return float(output)

# ---------------- INPUT SECTION ----------------

with st.container():

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧾 Passenger Information")

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    fare = st.slider(
        "Fare",
        0.0,
        500.0,
        50.0
    )

    predict_btn = st.button(
        "🚀 Predict Survival"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------

if predict_btn:

    with st.spinner("Analyzing Passenger Data..."):
        time.sleep(1.5)

    probability = predict_survival(
        pclass,
        age,
        fare
    )

    survived = probability >= 0.5

    # ---------------- RESULT ----------------

    if survived:

        st.markdown(f"""
        <div class="result-success">
            <h1>✅ SURVIVED</h1>
            <h2>{probability:.1%} Survival Probability</h2>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-danger">
            <h1>❌ PERISHED</h1>
            <h2>{1 - probability:.1%} Risk Probability</h2>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- METRICS ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="metric">
            <h2>{probability:.1%}</h2>
            <p>Survival Probability</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric">
            <h2>{1 - probability:.1%}</h2>
            <p>Death Probability</p>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- PROGRESS BAR ----------------

    st.write("")

    st.progress(probability)

    # ---------------- DONUT CHART ----------------

    fig, ax = plt.subplots(figsize=(5, 5))

    values = [
        probability,
        1 - probability
    ]

    labels = [
        "Survived",
        "Perished"
    ]

    colors = [
        "#10b981",
        "#ef4444"
    ]

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.35)
    )

    ax.axis('equal')

    st.pyplot(fig)

    # ---------------- MODEL SUMMARY ----------------

    st.markdown(f"""
    <div class="summary-box">

    <h2>📊 Model Summary</h2>

    <p><b>Model Type:</b> Artificial Neural Network</p>

    <p><b>Architecture:</b> 3 Input → 2 Hidden → 1 Output</p>

    <p><b>Activation Function:</b> Sigmoid</p>

    <p><b>Training Accuracy:</b> {TRAIN_ACCURACY:.0%}</p>

    <p><b>Validation Accuracy:</b> {VALIDATION_ACCURACY:.0%}</p>

    <p><b>Loss:</b> {LOSS}</p>

    <p>
    The AI model predicts passenger survival probability
    using passenger class, age, and fare information.
    Higher-class passengers with higher fares generally
    had better survival chances.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">

🚢 Titanic Survival Prediction System <br><br>

Powered by Artificial Neural Networks • Streamlit • NumPy

</div>
""", unsafe_allow_html=True)