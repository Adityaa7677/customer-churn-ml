import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('model.pkl')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    freq_flyer = st.selectbox("Frequent Flyer", options=["No", "Yes", "No Record"])
    income = st.selectbox("Annual Income Class", options=["Low Income", "Middle Income", "High Income"])

with col2:
    services = st.number_input("Services Opted", min_value=1, max_value=10, value=1)
    synced = st.selectbox("Social Media Synced", options=["No", "Yes"])
    hotel = st.selectbox("Booked Hotel Before", options=["No", "Yes"])

if st.button("Predict Churn Status"):
    
    f_flyer = 1 if freq_flyer == "Yes" else 0 
    inc_class = 0 if income == "High Income" else (1 if income == "Low Income" else 2)
    sync_val = 1 if synced == "Yes" else 0
    hotel_val = 1 if hotel == "Yes" else 0

    input_df = pd.DataFrame([[
        age, f_flyer, inc_class, services, sync_val, hotel_val
    ]], columns=['Age', 'FrequentFlyer', 'AnnualIncomeClass', 'ServicesOpted', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot'])

    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: High Churn Risk")
    else:
        st.success("### Result: Low Churn Risk (Loyal)")
