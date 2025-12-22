import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
st.set_page_config(
    page_title="Predicción de Precio de Autos",
    page_icon="🚗",
    layout="centered"
)

# ===============================
# CSS PERSONALIZADO
# ===============================
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #f9fafb;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }
    div[data-testid="stSidebar"] {
        background-color: #f3f4f6;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# CARGA DEL MODELO
# ===============================
@st.cache_resource
def load_objects():
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

model, model_columns = load_objects()

# ===============================
# HISTORIAL (SESSION STATE)
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# HEADER
# ===============================
st.title("🚗 Predicción de Precio de Autos")
st.markdown("Estimación del **precio de mercado** mediante *Machine Learning*.")

st.divider()

# ===============================
# SIDEBAR - INPUTS
# ===============================
st.sidebar.header("Características del vehículo")

# Autocompletar con selectbox
manufacturer = st.sidebar.selectbox(
    "Fabricante (buscar escribiendo)",
    sorted([
        "Toyota", "BMW", "Mercedes-Benz", "Audi", "Hyundai",
        "Kia", "Ford", "Chevrolet", "Nissan", "Honda",
        "Mazda", "Volkswagen", "Subaru", "Volvo", "Lexus"
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

has_turbo = st.sidebar.radio(
    "Turbo",
    ["Yes", "No"]
)
has_turbo = 1 if has_turbo == "Yes" else 0

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
    2, 5, 4
)

engine_volume = st.sidebar.selectbox(
    "Motor (L)",
    [0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
)

airbags = st.sidebar.number_input(
    "Número de airbags",
    0, 12, 2
)

mileage = st.sidebar.number_input(
    "Kilometraje",
    0, 500_000, 50_000, step=1_000
)

levy = st.sidebar.number_input(
    "Levy",
    0, 10_000, 0
)

prod_year = st.sidebar.number_input(
    "Año de producción",
    1980, 2025, 2018
)

car_age = 2025 - prod_year

# ===============================
# MONEDA (AL FINAL)
# ===============================
st.sidebar.divider()
show_mxn = st.sidebar.checkbox("Mostrar precio en MXN")

usd_to_mxn = 17.0

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
    "Airbags": airbags,
    "Has_Turbo": has_turbo,
    "Mileage": mileage,
    "Levy": levy,
    "Car_Age": car_age
}

df_input = pd.DataFrame([input_data])
df_input = pd.get_dummies(df_input)
df_input = df_input.reindex(columns=model_columns, fill_value=0)

# ===============================
# PREDICCIÓN
# ===============================
st.divider()

if st.button("🔮 Predecir precio"):
    price_usd = model.predict(df_input)[0]

    if show_mxn:
        price = price_usd * usd_to_mxn
        currency = "MXN"
    else:
        price = price_usd
        currency = "USD"

    st.subheader("Resultado de la predicción")
    st.metric(
        label=f"Precio estimado ({currency})",
        value=f"${price:,.0f}"
    )

    # Guardar en historial
    st.session_state.history.append({
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Fabricante": manufacturer,
        "Año": prod_year,
        "Precio": f"${price:,.0f}",
        "Moneda": currency
    })

# ===============================
# HISTORIAL DE PREDICCIONES
# ===============================
if st.session_state.history:
    st.divider()
    st.subheader("📊 Historial de predicciones")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.divider()
st.caption("Aplicación demostrativa para portafolio académico/profesional.")

