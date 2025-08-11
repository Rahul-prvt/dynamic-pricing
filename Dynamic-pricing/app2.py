import streamlit as st
import joblib

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        h1 {color: #1a237e; border-bottom: 2px solid #1a237e;}
        .sidebar .sidebar-content {background-color: #e8eaf6;}
        .stNumberInput, .stSelectbox {background-color: white;}
        .stButton>button {background-color: #1a237e; color: white; border-radius: 5px;}
        .prediction-box {background-color: white; padding: 20px; border-radius: 10px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_resources():
    return (
        joblib.load('model_hist.pkl'),
        joblib.load('model_adj.pkl'),
        joblib.load('scaler.pkl'),
        joblib.load('time_label.pkl'),
        joblib.load('vehicle_label.pkl')
    )

model_hist, model_adj, scaler, le_time, le_vehicle = load_resources()

# About Section in Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Fare Prediction System**  
    Advanced ML-powered solution for:
    - Real-time fare estimation
    - Dynamic pricing analysis
    - Operational profitability assessment

    **Technical Specifications:**  
    ▸ Historical Price Model: Random Forest
    ▸ Adjusted Price Model: Random Forest  
    ▸ Feature Scaling: StandardScaler  
    ▸ Version: 2.1.0  
    
    *Developed by Enterprise Analytics Team*  
    *Last updated: March 2025*
    """)

# Main Content
st.title("Ride Fare Analytics")
st.markdown("---")

# Input Form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        number_of_riders = st.number_input("Active Riders", min_value=0, step=1)
        number_of_drivers = st.number_input("Available Drivers", min_value=0, step=1)
    
    with col2:
        expected_duration = st.number_input("Duration (minutes)", min_value=0, step=1)
        time_of_booking = st.selectbox("Time Category", le_time.classes_)
        vehicle_type = st.selectbox("Vehicle Class", le_vehicle.classes_)

    submitted = st.form_submit_button("Generate Predictions")

# Prediction Logic
if submitted:
    try:
        # Feature processing
        time_encoded = le_time.transform([time_of_booking])[0]
        vehicle_encoded = le_vehicle.transform([vehicle_type])[0]

        features = [[number_of_riders, number_of_drivers, expected_duration,
                   time_encoded, vehicle_encoded]]
        features_scaled = scaler.transform(features)

        # Predictions
        actual_price = model_hist.predict(features_scaled)[0]
        adjusted_price = model_adj.predict(features_scaled)[0]
        is_profitable = adjusted_price > actual_price + 10

        # Display Results
        with st.container():
            st.markdown("### Prediction Results")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("**Base Fare**")
                st.markdown(f"<div class='prediction-box'>₹{actual_price:.2f}</div>", 
                           unsafe_allow_html=True)
            
            with col_b:
                st.markdown("**Adjusted Fare**")
                st.markdown(f"<div class='prediction-box'>₹{adjusted_price:.2f}</div>", 
                           unsafe_allow_html=True)
            
            with col_c:
                st.markdown("**Profitability**")
                status = "Profitable ✅" if is_profitable else "Not Viable ❌"
                st.markdown(f"<div class='prediction-box'>{status}</div>", 
                           unsafe_allow_html=True)

    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.markdown("*Please contact support if this error persists*")

st.markdown("---")
st.caption("Confidential Business Intelligence System - Authorized Use Only")