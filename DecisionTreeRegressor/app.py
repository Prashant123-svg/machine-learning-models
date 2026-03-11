import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🚗 Car Price Prediction App")

st.write("Enter the car details below to predict the selling price.")

# Input fields in columns
st.subheader("Car Details")
col1, col2 = st.columns(2)

# Car categories mapping
car_map = {
    "Maruti": 0,
    "Hyundai": 1,
    "Toyota": 2,
    "Ford": 3,
    "Honda": 4,
    "Skoda": 5,
    "MG": 6,
    "Mahindra": 7,
    "Tata": 8,
    "Renault": 9,
    "Kia": 10,
    "BMW": 11,
    "Audi": 12,
    "Mercedes": 13
}

with col1:
    car_name_str = st.selectbox("Car Brand", list(car_map.keys()))
    car_name = car_map[car_name_str]
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2025)
    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0)
    kms_driven = st.number_input("Kilometers Driven", min_value=0)

with col2:
    fuel_type = st.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG"))
    seller_type = st.selectbox("Seller Type", ("Dealer", "Individual"))
    transmission = st.selectbox("Transmission", ("Manual", "Automatic"))
    owner = st.selectbox("Number of Previous Owners", (0, 1, 2, 3))

# Encoding categorical inputs
fuel_map = {"Petrol":0, "Diesel":1, "CNG":2}
seller_map = {"Dealer":0, "Individual":1}
trans_map = {"Manual":0, "Automatic":1}

fuel_type = fuel_map[fuel_type]
seller_type = seller_map[seller_type]
transmission = trans_map[transmission]

# Display input summary
st.subheader("📋 Summary of Your Input")
details_col1, details_col2 = st.columns(2)

with details_col1:
    st.write(f"**Car Brand:** {car_name_str}")
    st.write(f"**Year:** {year}")
    st.write(f"**Present Price:** {present_price} Lakhs")
    st.write(f"**KM Driven:** {kms_driven}")

with details_col2:
    st.write(f"**Fuel Type:** {list(fuel_map.keys())[list(fuel_map.values()).index(int(fuel_type))]}")
    st.write(f"**Seller Type:** {list(seller_map.keys())[list(seller_map.values()).index(int(seller_type))]}")
    st.write(f"**Transmission:** {list(trans_map.keys())[list(trans_map.values()).index(int(transmission))]}")
    st.write(f"**Previous Owners:** {owner}")

st.divider()

# Prediction button
if st.button("Predict Selling Price"):

    input_data = np.array([[
        year,
        present_price,
        kms_driven,
        fuel_type,
        seller_type,
        transmission,
        owner
    ]])

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Selling Price: {prediction[0]:.2f} Lakhs")