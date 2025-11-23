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
# Label Encoding Maps
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

    # Missing flags
    input_data["missing_booking_value"] = 1 if input_data["Booking Value"] in [None, ""] else 0
    input_data["missing_payment_method"] = 1 if input_data["Payment Method"] in [None, ""] else 0
    input_data["missing_driver_rating"] = 1 if input_data["Driver Ratings"] in [None, ""] else 0
    input_data["missing_customer_rating"] = 1 if input_data["Customer Rating"] in [None, ""] else 0

    for col in ["Booking Value", "Customer Rating", "Driver Ratings"]:
        if input_data[col] in [None, ""]:
            input_data[col] = 0

    # Binary cancellation flags
    status = input_data["Booking Status"]
    input_data["is_cancelled_customer"] = 1 if status == "Cancelled by Customer" else 0
    input_data["is_cancelled_driver"] = 1 if status == "Cancelled by Driver" else 0
    input_data["is_incomplete"] = 1 if status == "Incomplete" else 0

    # Extract hour
    input_data["hour"] = input_data["Pickup Time"].hour

    # Label encode
    input_data["Booking Status"] = status_map[input_data["Booking Status"]]
    input_data["Payment Method"] = payment_map[input_data["Payment Method"]]

    # Final order
    ordered = [
        'Avg VTAT', 'Ride Distance', 'Booking Value',
        'Customer Rating', 'Driver Ratings',
        'Booking Status', 'Payment Method',
        'missing_booking_value', 'missing_payment_method',
        'missing_driver_rating', 'missing_customer_rating',
        'is_cancelled_customer', 'is_cancelled_driver', 'is_incomplete',
        'hour'
    ]

    X = pd.DataFrame([[input_data[col] for col in ordered]], columns=ordered)
    return X


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚕 Uber Ride Cancellation Prediction Model")
st.write("Provide ride details below to estimate cancellation probability.")

# Inputs
avg_vtat = st.number_input("Avg VTAT", min_value=0.0, max_value=120.0)
ride_distance = st.number_input("Ride Distance (KM)", min_value=0.0, max_value=500.0)
booking_value = st.number_input("Booking Value", min_value=0.0, max_value=10000.0)
customer_rating = st.number_input("Customer Rating", min_value=0.0, max_value=5.0)
driver_ratings = st.number_input("Driver Ratings", min_value=0.0, max_value=5.0)

booking_status = st.selectbox(
    "Booking Status",
    ["Completed", "Cancelled by Customer", "Cancelled by Driver", "Incomplete"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Cash", "Card", "Wallet", "UPI", "Other"]
)

# ----------------------------
# FIX: Time input no longer resets
# ----------------------------
if "default_pickup_time" not in st.session_state:
    st.session_state.default_pickup_time = datetime.now().time()

pickup_time = st.time_input(
    "Pickup Time",
    value=st.session_state.default_pickup_time
)

pickup_datetime = datetime.combine(datetime.today(), pickup_time)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict"):
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
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    st.subheader("🧾 Prediction Results")
    st.write(f"**Cancellation Probability:** {prob:.2%}")

    # -----------------------
    # FIX: Corrected logic
    # pred == 1 → Completed
    # pred == 0 → Cancelled
    # -----------------------
    if pred == 0:
        st.error("⚠️ High chance of cancellation.")
    else:
        st.success("✅ Low risk — booking likely to complete.")
