import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and feature column order
model = joblib.load("tuned_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # Must match training features

# Streamlit page settings
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("🔍 Bank Customer Churn Prediction")
st.markdown("Fill in the customer details below to check if they are likely to churn.")

# Input form
with st.form("churn_form"):
    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (Years with bank)", 0, 10, 3)
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    estimated_salary = st.number_input("Estimated Salary ($)", 10000.0, 200000.0, 80000.0, step=1000.0)
    
    gender = st.radio("Gender", ["Male", "Female"])
    geography = st.radio("Geography", ["France", "Germany", "Spain"])
    has_crcard = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

if submit:
    # 1. Feature Engineering
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'EstimatedSalary': estimated_salary,
        'Age_Tenure': age * tenure,
        'Balance_Salary_Ratio': balance / (estimated_salary + 1e-6),
        'CreditScore_Products': credit_score * num_products,
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'HasCrCard': 1 if has_crcard == "Yes" else 0,
        'IsActiveMember': 1 if is_active == "Yes" else 0
    }

    # 2. Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # 3. Ensure column order and presence matches training
    input_df = input_df.reindex(columns=feature_columns)

    # 4. Apply scaling to numerical columns only
    columns_to_scale = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
        'Age_Tenure', 'Balance_Salary_Ratio', 'CreditScore_Products'
    ]
    input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

    # 5. Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # 6. Output
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN with a probability of {probability:.2%}")
    else:
        st.success(f"✅ This customer is likely to STAY with a probability of {1 - probability:.2%}")
