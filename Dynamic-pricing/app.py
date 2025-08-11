import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, label encoders, and scaler
model = joblib.load("model.pkl")
label_encoders = joblib.load("label.pkl")
scaler = joblib.load("scalar.pkl")

# Streamlit UI
st.title("Dynamic Ride Pricing Prediction")
st.write("Enter ride details to predict the adjusted ride cost.")

# User inputs
expected_ride_duration = st.number_input("Expected Ride Duration (minutes)", min_value=1.0, step=0.1)
historical_cost = st.number_input("Historical Cost of Ride ($)", min_value=1.0, step=0.1)
demand_multiplier = st.slider("Demand Multiplier", min_value=0.5, max_value=2.0, step=0.1)
supply_multiplier = st.slider("Supply Multiplier", min_value=0.5, max_value=2.0, step=0.1)

# Ensure categorical features exist in label encoders
if "Vehicle_Type" in label_encoders and "Time_of_Booking" in label_encoders:
    vehicle_type = st.selectbox("Vehicle Type", list(label_encoders["Vehicle_Type"].classes_))
    time_of_booking = st.selectbox("Time of Booking", list(label_encoders["Time_of_Booking"].classes_))

    # Encode categorical features
    vehicle_type_encoded = label_encoders["Vehicle_Type"].transform([vehicle_type])[0]
    time_of_booking_encoded = label_encoders["Time_of_Booking"].transform([time_of_booking])[0]

    # Prepare input features
    input_data = np.array([
        [
            expected_ride_duration,
            historical_cost,
            demand_multiplier,
            supply_multiplier,
            vehicle_type_encoded,
            time_of_booking_encoded,
        ]
    ])

    # Scale all
    input_data = scaler.transform(input_data)

    # Predict adjusted ride cost
    if st.button("Predict Ride Cost"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Adjusted Ride Cost: ${prediction:.2f}")
else:
    st.error("Label encoders missing required categories. Ensure 'label.pkl' is correct.")
