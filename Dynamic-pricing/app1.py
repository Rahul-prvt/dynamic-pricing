import streamlit as st
import pandas as pd
import joblib

# Load models and encoders
model_hist = joblib.load('model_hist.pkl')
model_adj = joblib.load('model_adj.pkl')
scaler = joblib.load('scaler.pkl')
le_time = joblib.load('time_label.pkl')
le_vehicle = joblib.load('vehicle_label.pkl')

st.title("Ride Fare Prediction & Profitability Checker")

# Input form
with st.form("ride_input_form"):
    number_of_riders = st.number_input("Number of Riders", min_value=0, step=1)
    number_of_drivers = st.number_input("Number of Drivers", min_value=0, step=1)
    expected_duration = st.number_input("Expected Ride Duration (minutes)", min_value=0, step=1)
    
    time_of_booking = st.selectbox("Time of Booking", le_time.classes_)
    vehicle_type = st.selectbox("Vehicle Type", le_vehicle.classes_)

    submitted = st.form_submit_button("Predict Fare")

# Prediction logic
if submitted:
    try:
        # Encode categorical values
        time_encoded = le_time.transform([time_of_booking])[0]
        vehicle_encoded = le_vehicle.transform([vehicle_type])[0]

        # Prepare feature array
        features = [[
            number_of_riders,
            number_of_drivers,
            expected_duration,
            time_encoded,
            vehicle_encoded
        ]]

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict prices
        actual_price = model_hist.predict(features_scaled)[0]
        adjusted_price = model_adj.predict(features_scaled)[0]

        # Profit logic (example: profit if adjusted > actual + 10)
        profit_margin = 10
        is_profitable = adjusted_price > actual_price + profit_margin

        # Output
        st.subheader("Prediction Results")
        st.write(f"**Predicted Actual Price:** ₹{actual_price:.2f}")
        st.write(f"**Predicted Adjusted Price:** ₹{adjusted_price:.2f}")
        st.write(f"**Profitability:** {'✅ Profitable' if is_profitable else '❌ Not Profitable'}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
