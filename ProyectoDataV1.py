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
# CARGA DE OBJETOS
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
    Esta aplicación estima el **precio de mercado** de un automóvil
    utilizando un modelo de **Machine Learning (Random Forest)** entrenado
    con datos reales.
    """
)

st.divider()

# ===============================
# SIDEBAR - INPUTS
# ===============================
st.sidebar.header("Características del vehículo")

manufacturer = st.sidebar.text_input("Fabricante", "Toyota")
category = st.sidebar.selectbox("Categoría", ["Sedan", "SUV", "Hatchback"])
fuel_type = st.sidebar.selectbox("Tipo de combustible", ["Petrol", "Diesel", "Hybrid", "Electric"])
gear_box = st.sidebar.selectbox("Transmisión", ["Automatic", "Manual"])
drive_wheels = st.sidebar.selectbox("Tracción", ["Front", "Rear", "4x4"])
leather = st.sidebar.selectbox("Interior de cuero", ["Yes", "No"])
wheel = st.sidebar.selectbox("Volante", ["Left wheel", "Right wheel"])
color = st.sidebar.selectbox("Color", ["Black", "White", "Silver", "Gray"])
doors = st.sidebar.slider("Número de puertas", 2, 5, 4)
engine_volume = st.sidebar.slider("Motor (L)", 0.8, 6.0, 2.0)
has_turbo = st.sidebar.checkbox("Turbo")
mileage = st.sidebar.number_input("Kilometraje", 0, 500_000, 50_000)
levy = st.sidebar.number_input("Levy", 0, 10_000, 0)
car_age = st.sidebar.slider("Edad del auto (años)", 0, 30, 5)

# ===============================
# DATAFRAME DE ENTRADA
# ===============================
input_data = {
    "Manufacturer": manufacturer,
    "Category": category,
    "Fuel type": fuel_type,
    "Gear box type": gear_box,
    "Drive wheels": drive_wheels,
    "Leather interior": leather,
    "Wheel": wheel,
    "Color": color,
    "Doors": doors,
    "Engine volume": engine_volume,
    "Has_Turbo": int(has_turbo),
    "Mileage": mileage,
    "Levy": levy,
    "Car_Age": car_age,
}

df_input = pd.DataFrame([input_data])
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
    "Modelo entrenado con Random Forest. "
    "Los resultados son estimaciones y pueden variar según el mercado."
)
