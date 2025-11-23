import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open("urc_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ----------------------------
# Encoding Maps (MUST match training)
# ----------------------------
status_map = {
    "Completed": 0,
    "Cancelled by Customer": 1,
    "Cancelled by Driver": 2,
    "Incomplete": 3
}

payment_map = {
    "Cash": 0,
    "Card": 1,
    "Wallet": 2,
    "UPI": 3,
    "Other": 4
}

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess(input_data):

    # Missing-value flags
    input_data["missing_booking_value"] = int(input_data["Booking Value"] in [None, ""])
    input_data["missing_payment_method"] = int(input_data["Payment Method"] in [None, ""])
    input_data["missing_driver_rating"] = int(input_data["Driver Ratings"] in [None, ""])
    input_data["missing_customer_rating"] = int(input_data["Customer Rating"] in [None, ""])

    # Replace missing with zero
    for col in ["Booking Value", "Customer Rating", "Driver Ratings"]:
        if input_data[col] in [None, ""]:
            input_data[col] = 0

    # Binary flags for booking status
    status = input_data["Booking Status"]
    input_data["is_cancelled_customer"] = int(status == "Cancelled by Customer")
    input_data["is_cancelled_driver"] = int(status == "Cancelled by Driver")
    input_data["is_incomplete"] = int(status == "Incomplete")

    # Extract hour
    input_data["hour"] = input_data["Pickup Time"].hour

    # Encode categorical values
    input_data["Booking Status"] = status_map[input_data["Booking Status"]]
    input_data["Payment Method"] = payment_map[input_data["Payment Method"]]

    # Feature order
    ordered = [
        'Avg VTAT', 'Ride Distance', 'Booking Value',
        'Customer Rating', 'Driver Ratings',
        'Booking Status', 'Payment Method',
        'missing_booking_value', 'missing_payment_method',
        'missing_driver_rating', 'missing_customer_rating',
        'is_cancelled_customer', 'is_cancelled_driver', 'is_incomplete',
        'hour'
    ]

    return pd.DataFrame([[input_data[col] for col in ordered]], columns=ordered)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Uber Cancellation Predictor", page_icon="🚕", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center;'>🚕 Uber Ride Cancellation Prediction</h1>
    <p style='text-align:center;font-size:18px;color:#444;'>
        Enter ride details below to estimate the probability of cancellation.
    </p>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🔮 Prediction", "ℹ️ Model Info"])

# ----------------------------------
# TAB 1 — Prediction Interface
# ----------------------------------
with tab1:

    st.subheader("📋 Enter Ride Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_vtat = st.number_input("Avg VTAT", min_value=0.0, max_value=120.0)
        booking_value = st.number_input("Booking Value", min_value=0.0, max_value=10000.0)
        customer_rating = st.number_input("Customer Rating", min_value=0.0, max_value=5.0)

    with col2:
        ride_distance = st.number_input("Ride Distance (km)", min_value=0.0, max_value=500.0)
        driver_ratings = st.number_input("Driver Ratings", min_value=0.0, max_value=5.0)
        booking_status = st.selectbox(
            "Booking Status",
            ["Completed", "Cancelled by Customer", "Cancelled by Driver", "Incomplete"]
        )

    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            ["Cash", "Card", "Wallet", "UPI", "Other"]
        )
        pickup_time = st.time_input("Pickup Time", value=datetime.now().time())

    pickup_datetime = datetime.combine(datetime.today(), pickup_time)

    st.markdown("---")

    if st.button("🚀 Predict Cancellation", use_container_width=True):
        data = {
            "Avg VTAT": avg_vtat,
            "Ride Distance": ride_distance,
            "Booking Value": booking_value,
            "Customer Rating": customer_rating,
            "Driver Ratings": driver_ratings,
            "Booking Status": booking_status,
            "Payment Method": payment_method,
            "Pickup Time": pickup_datetime
        }

        X = preprocess(data)
        prob = model.predict_proba(X)[0][1]   # Probability of COMPLETED
        pred = model.predict(X)[0]            # 1 = completed, 0 = cancelled

        st.subheader("📊 Prediction Results")

        # ------------------------
        # Gauge-style probability
        # ------------------------
        st.metric(
            label="Probability Ride Will Be Completed",
            value=f"{prob:.2%}"
        )

        # Corrected prediction logic
        if pred == 1:
            st.success("✅ Low risk — booking likely to complete.")
        else:
            st.error("⚠️ High chance of cancellation.")

# ----------------------------------
# TAB 2 — Model Info
# ----------------------------------
with tab2:
    st.subheader("ℹ️ About This Model")
    st.write("""
    This Uber ride cancellation prediction model analyzes:
    - Customer & driver ratings  
    - Travel distance  
    - VTAT (Vehicle Time Arrival Time)  
    - Payment method  
    - Booking status  
    - Time of day  
      
    The model is trained using **XGBoost** and outputs:  
    - **Probability of ride being completed**  
    - A binary prediction (completed vs cancelled)  
    """)

    st.success("Prediction logic fixed: Model now interprets 1 = Completed, 0 = Cancelled correctly.")
