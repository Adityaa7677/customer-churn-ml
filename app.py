import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model
# Ensure the file on GitHub is exactly named 'model.pkl'
model = joblib.load('model.pkl')

# 2. Page Setup
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn risk.")

# 3. Inputs (Matching your CSV exactly) [cite: 150-155]
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    freq_flyer = st.selectbox("Frequent Flyer", options=["No", "Yes", "No Record"])
    income = st.selectbox("Annual Income Class", options=["Low Income", "Middle Income", "High Income"])

with col2:
    services = st.number_input("Services Opted", min_value=1, max_value=10, value=1)
    synced = st.selectbox("Social Media Synced", options=["No", "Yes"])
    hotel = st.selectbox("Booked Hotel Before", options=["No", "Yes"])

# 4. Prediction Logic
if st.button("Predict Churn Status"):
    # Encoding to match the Random Forest training [cite: 48-51]
    # Mapping based on typical LabelEncoder alphabetical sorting
    # FrequentFlyer: No=0, Yes=1 (No Record mapped to No)
    # Income: High=0, Low=1, Middle=2
    # Binary: No=0, Yes=1
    
    f_flyer = 1 if freq_flyer == "Yes" else 0 
    inc_class = 0 if income == "High Income" else (1 if income == "Low Income" else 2)
    sync_val = 1 if synced == "Yes" else 0
    hotel_val = 1 if hotel == "Yes" else 0

    # Create input dataframe with EXACT column names from CSV [cite: 43-44]
    input_df = pd.DataFrame([[
        age, f_flyer, inc_class, services, sync_val, hotel_val
    ]], columns=['Age', 'FrequentFlyer', 'AnnualIncomeClass', 'ServicesOpted', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot'])

    # Get Prediction
    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: High Churn Risk")
    else:
        st.success("### Result: Low Churn Risk (Loyal)")
