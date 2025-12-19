import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# CARGA DE OBJETOS
# ======================
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Predicción de Precio de Automóviles")

st.write("Ingrese las características del vehículo:")

# ======================
# INPUTS DEL USUARIO
# ======================
manufacturer = st.text_input("Manufacturer", "Toyota")
category = st.selectbox("Category", ["Sedan", "SUV", "Hatchback", "Coupe", "Universal"])
fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "Hybrid", "Electric"])
gear = st.selectbox("Gear box type", ["Automatic", "Manual"])
drive = st.selectbox("Drive wheels", ["Front", "Rear", "4x4"])
leather = st.selectbox("Leather interior", ["Yes", "No"])
wheel = st.selectbox("Wheel", ["Left wheel", "Right wheel"])
color = st.selectbox("Color", ["Black", "White", "Silver", "Blue", "Red"])
doors = st.number_input("Doors", 2, 5, 4)
engine_volume = st.number_input("Engine volume (L)", 0.8, 6.0, 2.0)
turbo = st.selectbox("Has Turbo", ["No", "Yes"])
mileage = st.number_input("Mileage (km)", 0, 500000, 60000)
levy = st.number_input("Levy", 0, 5000, 0)
year = st.number_input("Production Year", 1990, 2025, 2018)

# ======================
# PREDICCIÓN
# ======================
if st.button("Predecir precio"):
    data = {
        "Manufacturer": manufacturer,
        "Category": category,
        "Fuel type": fuel,
        "Gear box type": gear,
        "Drive wheels": drive,
        "Leather interior": leather,
        "Wheel": wheel,
        "Color": color,
        "Doors": doors,
        "Engine volume": engine_volume,
        "Has_Turbo": 1 if turbo == "Yes" else 0,
        "Mileage": mileage,
        "Levy": levy,
        "Car_Age": 2025 - year,
        "Model_Grouped": "Other"
    }

    df_input = pd.DataFrame([data])

    # Encoding
    df_input["Manufacturer"] = le.transform(df_input["Manufacturer"])

    df_input = pd.get_dummies(df_input)

    # Alinear columnas con el modelo
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df_input)

    st.success(f"Precio estimado: ${prediction[0]:,.0f}")
