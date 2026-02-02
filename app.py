import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
.card {
    background-color: #262730;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}
div.stButton > button {
    width: 100%;
    height: 45px;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("📌 Customer Churn App")

st.sidebar.markdown("""
**🔍 What does this app do?**  
This application predicts whether a bank customer is likely to leave the bank.

**🤖 Model Used**  
Artificial Neural Network (ANN)

**📊 Input Features**
- Credit Score  
- Age  
- Balance  
- Geography  
- Products  
- Activity Status  

**⚙️ How to Use**
1. Enter customer details  
2. Click **Predict Churn**  
3. View probability & result  

---

**👨‍💻 Developer**
Ujjwal  
ML + Full Stack Project
""")

st.sidebar.markdown("🧠 *Built using Streamlit & TensorFlow*")

# ================= LOAD MODEL & FILES =================
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ================= HEADER =================
st.markdown("""
<h1 style='text-align:center; color:#6C63FF;'>
Customer Churn Prediction
</h1>
<p style='text-align:center; font-size:18px;'>
ANN based Machine Learning Web App
</p>
""", unsafe_allow_html=True)

# ================= INPUT CARD =================
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92)
    tenure = st.slider("📆 Tenure (years)", 0, 10)

with col2:
    credit_score = st.number_input("📊 Credit Score", min_value=300, max_value=900)
    balance = st.number_input("💰 Balance")
    num_of_products = st.slider("📦 Number of Products", 1, 4)
    estimated_salary = st.number_input("💼 Estimated Salary")

has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("⚡ Is Active Member", [0, 1])

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICT BUTTON =================
if st.button("🔮 Predict Churn"):

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
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    final_input = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    final_input_scaled = scaler.transform(final_input)

    prediction = model.predict(final_input_scaled)
    prediction_proba = prediction[0][0]

    # ================= RESULT CARD =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📈 Prediction Result")
    st.progress(float(prediction_proba))
    st.write(f"**Churn Probability:** `{prediction_proba:.2f}`")

    if prediction_proba > 0.5:
        st.error("❌ The customer is likely to churn")
    else:
        st.success("✅ The customer is not likely to churn")

    st.markdown("</div>", unsafe_allow_html=True)
