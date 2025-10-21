import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================
# Load model, scaler, encoders
# ==========================
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# ==========================
# Streamlit UI setup
# ==========================
st.set_page_config(page_title="🏡 House Price Prediction App", page_icon="🏡")
st.title("🏡 House Price Prediction App")

st.subheader("🔹 Enter House Features")

# ==========================
# Inputs
# ==========================
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
garage_cars = st.slider("Number of Garage Cars", 0, 10, 1)

values = {
    "No" : "N",
    "Yes" : "Y"
}
central_air_choice = st.selectbox("Central Air", list(values.keys()))
central_air = values[central_air_choice]

values = {
    "Excellent" : "Ex",
    "Good" : "Gd",
    "Typical/Average" : "TA",
    "Fair" : "Fa",
    "Poor" : "Po", 
}
bsmt_qual_choice = st.selectbox("Basement Quality", list(values.keys()))
bsmt_qual = values[bsmt_qual_choice]

gr_liv_area = st.number_input("Above Grade Living Area (sqft)", 300, 10000, 1200)

values = {
    "Attached" : "Attchd",
    "Detached" : "Detchd",
    "Built-In" : "BuiltIn",
    "Carport" : "CarPort",
    "Two Types" : "2Types"
}
garage_type_choice = st.selectbox("Garage Type", list(values.keys()))
garage_type = values[garage_type_choice]

values = {
    "Excellent" : "Ex",
    "Good" : "Gd",
    "Typical/Average" : "TA",
    "Fair" : "Fa",
}
kitchen_qual_choice = st.selectbox("Kitchen Quality", list(values.keys()))
kitchen_qual = values[kitchen_qual_choice]

total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 5000, 800)
fireplaces = st.slider("Number of Fireplaces", 0, 5, 1)

values = {
    "Level" : "Lvl",
    "Banked" : "Bnk",
    "Hillside" : "HLS",
    "Low" : "Low"
}
land_contour_choice = st.selectbox("Land Contour Type", list(values.keys()))
land_contour = values[land_contour_choice]

first_flr_sf = st.number_input("First Floor Area (sqft)", 300, 5000, 1000)

# ==========================
# DataFrame for model input
# ==========================
input_data = pd.DataFrame({
    'Overall Qual': [overall_qual],
    'Garage Cars': [garage_cars],
    'Central Air': [central_air],
    'Bsmt Qual': [bsmt_qual],
    'Gr Liv Area': [gr_liv_area],
    'Garage Type': [garage_type],
    'Kitchen Qual': [kitchen_qual],
    'Total Bsmt SF': [total_bsmt_sf],
    'Fireplaces': [fireplaces],
    'Land Contour': [land_contour],
    '1st Flr SF': [first_flr_sf]
})

# ==========================
# Apply LabelEncoders
# ==========================
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

# ==========================
# Apply Scaler and Predict
# ==========================
scaled_data = scaler.transform(input_data)

if st.button("🔍 Predict Price"):
    log_pred = model.predict(scaled_data)
    prediction = np.expm1(log_pred)  # Inverse log1p to return the true price
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")