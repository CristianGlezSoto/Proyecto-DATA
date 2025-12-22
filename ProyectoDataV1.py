import streamlit as st
import pandas as pd
import joblib

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
st.set_page_config(
    page_title="Predicción de Precio de Autos",
    page_icon="🚗",
    layout="centered"
)

# ===============================
# CARGA DE MODELO Y COLUMNAS
# ===============================
@st.cache_resource
def load_objects():
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

model, model_columns = load_objects()

# ===============================
# HEADER
# ===============================
st.title("🚗 Predicción de Precio de Autos")
st.markdown(
    """
    Estima el **precio de mercado** de un automóvil utilizando un modelo
    de **Machine Learning (Random Forest)** entrenado con datos reales.
    """
)

st.divider()

# ===============================
# SIDEBAR - INPUTS
# ===============================
st.sidebar.header("Características del vehículo")

manufacturer = st.sidebar.selectbox(
    "Fabricante",
    sorted([
        "Toyota", "BMW", "Mercedes-Benz", "Audi", "Hyundai",
        "Kia", "Ford", "Chevrolet", "Nissan", "Honda"
    ])
)

category = st.sidebar.selectbox(
    "Categoría",
    ["Sedan", "SUV", "Hatchback", "Coupe", "Universal"]
)

fuel_type = st.sidebar.selectbox(
    "Tipo de combustible",
    ["Petrol", "Diesel", "Hybrid", "Electric"]
)

# -------------------------------
# OPCIONES BINARIAS (RADIO)
# -------------------------------
gear_box = st.sidebar.radio(
    "Transmisión",
    ["Automatic", "Manual"]
)

leather_interior = st.sidebar.radio(
    "Interior de cuero",
    ["Yes", "No"]
)

wheel = st.sidebar.radio(
    "Tipo de volante",
    ["Left wheel", "Right wheel"]
)

has_turbo_option = st.sidebar.radio(
    "Turbo",
    ["Yes", "No"]
)
has_turbo = 1 if has_turbo_option == "Yes" else 0

# -------------------------------
# RESTO DE VARIABLES
# -------------------------------
drive_wheels = st.sidebar.selectbox(
    "Tracción",
    ["Front", "Rear", "4x4"]
)

color = st.sidebar.selectbox(
    "Color",
    ["Black", "White", "Silver", "Gray", "Blue", "Red"]
)

doors = st.sidebar.number_input(
    "Número de puertas",
    min_value=2,
    max_value=5,
    value=4,
    step=1
)

engine_volume = st.sidebar.selectbox(
    "Motor (L)",
    [0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
)

mileage = st.sidebar.number_input(
    "Kilometraje",
    min_value=0,
    max_value=500_000,
    value=50_000,
    step=1_000
)

levy = st.sidebar.number_input(
    "Levy",
    min_value=0,
    max_value=10_000,
    value=0
)

prod_year = st.sidebar.number_input(
    "Año de producción",
    min_value=1980,
    max_value=2025,
    value=2018,
    step=1
)

car_age = 2025 - prod_year

# ===============================
# DATAFRAME DE ENTRADA
# ===============================
input_data = {
    "Manufacturer": manufacturer,
    "Category": category,
    "Fuel type": fuel_type,
    "Gear box type": gear_box,
    "Drive wheels": drive_wheels,
    "Leather interior": leather_interior,
    "Wheel": wheel,
    "Color": color,
    "Doors": doors,
    "Engine volume": engine_volume,
    "Has_Turbo": has_turbo,
    "Mileage": mileage,
    "Levy": levy,
    "Car_Age": car_age
}

df_input = pd.DataFrame([input_data])

# One-hot encoding
df_input = pd.get_dummies(df_input)

# Alinear columnas con el modelo
df_input = df_input.reindex(columns=model_columns, fill_value=0)

# ===============================
# PREDICCIÓN
# ===============================
st.divider()

if st.button("🔮 Predecir precio"):
    prediction = model.predict(df_input)[0]

    st.subheader("Resultado de la predicción")
    st.metric(
        label="Precio estimado",
        value=f"${prediction:,.0f}"
    )

# ===============================
# FOOTER
# ===============================
st.divider()
st.caption(
    "Modelo Random Forest. El precio estimado es una aproximación basada en patrones históricos."
)

