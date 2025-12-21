import streamlit as st
import pandas as pd
import joblib

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

st.title("🚗 Predicción de Precio de Automóviles")
st.write(
    "Esta aplicación estima el precio de un automóvil usando un modelo "
    "Random Forest entrenado con datos reales."
)

# ======================================================
# CARGA DE OBJETOS DEL MODELO
# ======================================================
@st.cache_resource
def load_objects():
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, le, columns

model, le, model_columns = load_objects()

# ======================================================
# INTERFAZ DE ENTRADA
# ======================================================
st.header("📋 Características del vehículo")

col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox(
        "Manufacturer",
        sorted(le.classes_)
    )

    category = st.selectbox(
        "Category",
        ["Sedan", "SUV", "Hatchback", "Coupe", "Universal"]
    )

    fuel_type = st.selectbox(
        "Fuel type",
        ["Petrol", "Diesel", "Hybrid", "Electric"]
    )

    gearbox = st.selectbox(
        "Gear box type",
        ["Automatic", "Manual"]
    )

    drive = st.selectbox(
        "Drive wheels",
        ["Front", "Rear", "4x4"]
    )

    leather = st.selectbox(
        "Leather interior",
        ["Yes", "No"]
    )

with col2:
    engine_volume = st.slider(
        "Engine volume (L)",
        min_value=0.8,
        max_value=6.0,
        value=2.0,
        step=0.1
    )

    has_turbo = st.selectbox(
        "Turbo",
        ["No", "Yes"]
    )

    mileage = st.slider(
        "Mileage (km)",
        0,
        500_000,
        60_000,
        step=1_000
    )

    doors = st.selectbox(
        "Doors",
        [2, 3, 4, 5]
    )

    year = st.slider(
        "Production year",
        1990,
        2025,
        2018
    )

    levy = st.number_input(
        "Levy",
        min_value=0,
        max_value=10_000,
        value=0
    )

    color = st.selectbox(
        "Color",
        ["Black", "White", "Silver", "Blue", "Red", "Grey", "Other"]
    )

# ======================================================
# BOTÓN DE PREDICCIÓN
# ======================================================
st.markdown("---")

if st.button("💰 Predecir precio", use_container_width=True):

    # Construcción del DataFrame de entrada
    input_data = {
        "Manufacturer": manufacturer,
        "Category": category,
        "Fuel type": fuel_type,
        "Gear box type": gearbox,
        "Drive wheels": drive,
        "Leather interior": leather,
        "Wheel": "Left wheel",
        "Color": color,
        "Doors": doors,
        "Engine volume": engine_volume,
        "Has_Turbo": 1 if has_turbo == "Yes" else 0,
        "Mileage": mileage,
        "Levy": levy,
        "Car_Age": 2025 - year,
        "Model_Grouped": "Other"
    }

    df_input = pd.DataFrame([input_data])

    # ==================================================
    # PREPROCESAMIENTO
    # ==================================================
    df_input["Manufacturer"] = le.transform(df_input["Manufacturer"])

    df_input = pd.get_dummies(df_input)

    # Alinear columnas exactamente como el modelo
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # ==================================================
    # PREDICCIÓN
    # ==================================================
    prediction = model.predict(df_input)[0]

    st.success(f"💵 Precio estimado: **${prediction:,.0f}**")
