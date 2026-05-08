import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Churn Dashboard", layout="wide")

# ------------------ MODERN UI CSS ------------------
st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #111827;
}

/* ===== MAIN TITLE ===== */
h1 {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #6366f1, #06b6d4, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}

/* ===== SUBTEXT ===== */
p {
    text-align: center;
    color: #6b7280;
    font-size: 16px;
}

/* ===== GLASS DASHBOARD CARD ===== */
.block-container {
    padding: 2rem 3rem;
}

/* Input cards feel */
.stSelectbox, .stNumberInput, .stSlider {
    background: white !important;
    border-radius: 14px;
    padding: 10px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    border: 1px solid #f1f5f9;
}

/* Labels */
label {
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ===== BUTTON (MODERN SAAS STYLE) ===== */
.stButton > button {
    width: 100%;
    padding: 0.9rem;
    border-radius: 14px;
    border: none;
    font-size: 18px;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    box-shadow: 0 10px 25px rgba(99,102,241,0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(99,102,241,0.35);
}

/* ===== RESULT CARD ===== */
.result {
    margin-top: 25px;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    background: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 1px solid #f3f4f6;
}

/* success glow */
.success-box {
    color: #16a34a;
}

/* error glow */
.error-box {
    color: #dc2626;
}

/* ===== SIDEBAR (if used later) ===== */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)
# ------------------ TITLE ------------------
st.markdown("<h1>AI Customer Churn Dashboard</h1>", unsafe_allow_html=True)

# ------------------ LAYOUT ------------------
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92)
    balance = st.number_input('💰 Balance')

with col2:
    credit_score = st.number_input('📊 Credit Score')
    estimated_salary = st.number_input('💵 Estimated Salary')
    tenure = st.slider('📅 Tenure', 0, 10)
    num_of_products = st.slider('📦 Number of Products', 1, 4)

has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('⚡ Active Member', [0, 1])

# ------------------ PREDICTION ------------------
if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data, geo_df], axis=1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0][0]

    st.markdown(f"""
        <div class="result">
            🔮 Churn Probability: {prediction:.2f}
        </div>
    """, unsafe_allow_html=True)

    if prediction > 0.5:
        st.error("⚠️ High Risk: Customer likely to churn")
    else:
        st.success("✅ Low Risk: Customer likely to stay")