import streamlit as st
import pandas as pd
import pickle

# Load model
with open("myntra_price_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("myntra_price_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("myntra_price_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("🛍️ Myntra Pants Price Predictor")

# ---- NUMERICAL INPUTS ----
MRP               = st.number_input("MRP (₹)",           min_value=499,   max_value=72000, value=2999)
discount_percent  = st.number_input("Discount (%)",       min_value=0.0,   max_value=64.0,  value=10.0, step=0.5)
ratings           = st.number_input("Ratings",            min_value=1.0,   max_value=5.0,   value=4.0,  step=0.1)
number_of_ratings = st.number_input("Number of Ratings",  min_value=5,     max_value=30700, value=100)

# ---- CATEGORICAL INPUTS ----
brand_name  = st.selectbox("Brand Name", sorted(encoders["brand_name"].classes_.tolist()))
fit_type    = st.selectbox("Fit Type",   ["Slim","Loose","Regular","Relaxed","Skinny","Wide","Straight","Other"])
fabric_type = st.selectbox("Fabric Type",["Denim","Cotton","Linen","Polyester","Other"])

# ---- CREATE INPUT DATAFRAME ----
input_data = pd.DataFrame({
    "brand_name"        : [brand_name],
    "MRP"               : [MRP],
    "discount_percent"  : [discount_percent],
    "ratings"           : [ratings],
    "number_of_ratings" : [number_of_ratings],
    "fit_type"          : [fit_type],
    "fabric_type"       : [fabric_type],
})

# ---- PREDICTION ----
if st.button("Predict Price"):

    # Encode categoricals
    for col in ["brand_name", "fit_type", "fabric_type"]:
        le  = encoders[col]
        val = input_data[col].iloc[0]
        input_data[col] = le.transform([val]) if val in le.classes_ else [0]

    # Scale and predict
    input_scaled    = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]

    st.success(f"💰 Predicted Price: ₹ {predicted_price:,.0f}")
