import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="🏡 House Price Prediction App", page_icon="🏡")
st.title("🏡 House Price Prediction App")
st.subheader("🔹 Enter House Features")

overall_qual = st.slider("Overall Quality", 2, 10, 2)
gr_liv_area = st.number_input("Above Ground Living Area (334 - 2667) (sqft)", 334, 2667, 334)
garage_cars = st.slider("Number of Garage Cars", 0, 3, 0)
garage_area = st.number_input("Garage Area (0 - 960) (sqft)", 0, 960, 0)
total_bs_sf = st.number_input("Total Basement Area (30 - 2064) (sqft)", 30, 2064, 30)
st1_flr_sf = st.number_input("First Floor Area (334 - 2145) (sqft)", 334, 2145, 334)
year_built = st.number_input("Construction Year (1883 - 2010)", 1883, 2010, 1883)
full_bath = st.number_input("Number of Bathrooms (0 - 3)", 0, 3, 0)
year_remod_or_add = st.number_input("Remodel Year (1950 - 2010)", 1950, 2010, 1950)
garage_year_blt = st.number_input("Garage Build Year (1904 - 2059)", 1904, 2059, 1904)
totrms_abvgrd = st.slider("Total Rooms Above Ground", 2, 10, 2)
fireplaces = st.slider("Number of Fireplaces", 0, 2, 0)
mas_vnr_area = st.number_input("Masonry Veneer Area (0 - 406) (sqft)", 0, 406, 0)
foundation = st.selectbox("Foundation Type", encoders['Foundation'].classes_)
bsmtfin_sf1 = st.number_input("Basement Finished Area 1  (0 - 1835) (sqft)", 0, 1835, 0)

input_data = pd.DataFrame({
    'Overall Qual': [overall_qual],
    'Gr Liv Area': [gr_liv_area],
    'Garage Cars': [garage_cars],
    'Garage Area': [garage_area],
    'Total Bsmt SF': [total_bs_sf],
    '1st Flr SF': [st1_flr_sf],
    'Year Built': [year_built],
    'Full Bath': [full_bath],
    'Year Remod/Add': [year_remod_or_add],
    'Garage Yr Blt': [garage_year_blt],
    'TotRms AbvGrd': [totrms_abvgrd],
    'Fireplaces': [fireplaces],
    'Mas Vnr Area': [mas_vnr_area],
    'Foundation': [foundation],
    'BsmtFin SF 1': [bsmtfin_sf1]
})

for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

scaled_data = scaler.transform(input_data)

if st.button("🔍 Predict Price"):
    log_pred = model.predict(scaled_data)
    prediction = np.expm1(log_pred)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
