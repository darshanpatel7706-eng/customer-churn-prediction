import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# Model aur explainer load karo
model = joblib.load('model.pkl')
explainer = joblib.load('explainer.pkl')

with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

st.title("🔮 Customer Churn Predictor")
st.write("Customer ki details bharo aur jaano — kya woh company chhod sakta hai?")

# Sidebar inputs
st.sidebar.header("Customer Details bharein")

tenure = st.sidebar.slider("Kitne mahine se customer hai?", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0, 150, 70)
total_charges = st.sidebar.slider("Total Charges ($)", 0, 9000, 1000)

contract = st.sidebar.selectbox("Contract Type",
    ["Month-to-month", "One year", "Two year"])

internet = st.sidebar.selectbox("Internet Service",
    ["DSL", "Fiber optic", "No"])

senior = st.sidebar.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner hai?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents hain?", ["No", "Yes"])
paperless = st.sidebar.selectbox("Paperless Billing?", ["No", "Yes"])

# Input data banao
input_dict = {feature: 0 for feature in feature_names}

input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges
input_dict['SeniorCitizen'] = 1 if senior == "Yes" else 0
input_dict['Partner'] = 1 if partner == "Yes" else 0
input_dict['Dependents'] = 1 if dependents == "Yes" else 0
input_dict['PaperlessBilling'] = 1 if paperless == "Yes" else 0

if contract == "One year":
    input_dict['Contract_One year'] = 1
elif contract == "Two year":
    input_dict['Contract_Two year'] = 1

if internet == "Fiber optic":
    input_dict['InternetService_Fiber optic'] = 1
elif internet == "No":
    input_dict['InternetService_No'] = 1

input_df = pd.DataFrame([input_dict])

# Predict button
if st.button("🔮 Predict Karo"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer CHURN karega! ({probability*100:.1f}% probability)")
    else:
        st.success(f"✅ Customer STAY karega! ({probability*100:.1f}% churn probability)")

    # SHAP Graph
    st.subheader("🔍 Kyun yeh prediction?")
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[:,:,1][0],
            base_values=explainer.expected_value[1],
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)