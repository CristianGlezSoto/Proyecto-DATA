import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
st.set_page_config(
    page_title="Predicción de Precio de Autos",
    page_icon="🚗",
    layout="centered"
)

# ======================================================
# CSS PERSONALIZADO
# ======================================================
st.markdown(
    """
    <style>
    body {
        background-color: #f8fafc;
    }
    div[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        padding: 20px;
    }
    .result-card {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        padding: 35px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-top: 25px;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    }
    .result-card h1 {
        margin: 10px 0;
        font-size: 42px;
    }
    .result-card p {
        opacity: 0.85;
        margin: 0;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 12px;
        height: 3em;
        font-size: 16px;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# CARGA DE MODELO
# ======================================================
@st.cache_resource
def load_objects():
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

model, model_columns = load_objects()

# ======================================================
# HISTORIAL
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# HEADER
# ======================================================
st.title("🚗 Predicción de Precio de Autos")
st.markdown(
    "Estimación del **precio de mercado** basada en un modelo "
    "**Random Forest** entrenado con datos históricos."
)

st.divider()

# ======================================================
# SIDEBAR - INPUTS
# ======================================================
st.sidebar.title("Configuración del vehículo")

with st.sidebar.expander("🚘 Información general", expanded=True):
    manufacturer = st.selectbox(
        "Fabricante",
        sorted([
            "Toyota", "BMW", "Mercedes-Benz", "Audi", "Hyundai",
            "Kia", "Ford", "Chevrolet", "Nissan", "Honda",
            "Mazda", "Volkswagen", "Subaru", "Volvo", "Lexus"
        ])
    )

    category = st.selectbox(
        "Categoría",
        ["Sedan", "SUV", "Hatchback", "Coupe", "Universal"]
    )

    prod_year = st.number_input(
        "Año de producción",
        min_value=1980,
        max_value=2025,
        value=2018
    )

with st.sidebar.expander("⚙️ Motor y desempeño"):
    engine_volume = st.selectbox(
        "Motor (litros)",
        [0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    )

    gear_box = st.radio(
        "Transmisión",
        ["Automatic", "Manual"]
    )

    fuel_type = st.selectbox(
        "Tipo de combustible",
        ["Petrol", "Diesel", "Hybrid", "Electric"]
    )

    drive_wheels = st.selectbox(
        "Tracción",
        ["Front", "Rear", "4x4"]
    )

    mileage = st.number_input(
        "Kilometraje total",
        min_value=0,
        max_value=500_000,
        value=50_000,
        step=1_000
    )

    has_turbo = st.radio(
        "Turbo",
        ["Yes", "No"]
    )

with st.sidebar.expander("🛡️ Diseño y seguridad"):
    leather_interior = st.radio(
        "Interior de cuero",
        ["Yes", "No"]
    )

    airbags = st.number_input(
        "Número de airbags",
        min_value=0,
        max_value=12,
        value=2
    )

    color = st.selectbox(
        "Color",
        ["Black", "White", "Silver", "Gray", "Blue", "Red"]
    )

    wheel = st.radio(
        "Volante",
        ["Left wheel", "Right wheel"]
    )

    doors = st.number_input(
        "Número de puertas",
        min_value=2,
        max_value=5,
        value=4
    )
    
# ======================================================
# MONEDA
# ======================================================
st.sidebar.divider()
show_mxn = st.sidebar.checkbox("Mostrar precio en MXN")
usd_to_mxn = 17.0

# ======================================================
# PREPARACIÓN DE DATOS
# ======================================================
car_age = 2025 - prod_year

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
    "Has_Turbo": 1 if has_turbo == "Yes" else 0,
    "Mileage": mileage,
    "Car_Age": car_age
}

df_input = pd.DataFrame([input_data])
df_input = pd.get_dummies(df_input)
df_input = df_input.reindex(columns=model_columns, fill_value=0)

# ======================================================
# PREDICCIÓN
# ======================================================
st.divider()

if st.button("Calcular precio estimado"):
    price_usd = model.predict(df_input)[0]

    if show_mxn:
        price = price_usd * usd_to_mxn
        currency = "MXN"
    else:
        price = price_usd
        currency = "USD"

    st.markdown(
        f"""
        <div class="result-card">
            <p>Precio estimado ({currency})</p>
            <h1>${price:,.0f}</h1>
            <p>Estimación basada en datos históricos</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.history.append({
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Fabricante": manufacturer,
        "Año": prod_year,
        "Precio": f"${price:,.0f}",
        "Moneda": currency
    })

# ======================================================
# HISTORIAL
# ======================================================
if st.session_state.history:
    st.divider()
    col1, col2 = st.columns([4, 1])

    with col1:
        st.subheader("📊 Historial de predicciones")
    with col2:
        if st.button("🗑️ Limpiar"):
            st.session_state.history = []

    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True
    )

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(
    "Aplicación demostrativa con fines académicos y de portafolio. Desarrollado por: Angel Jared Fitch Machado, Cristian Gonzalez Soto y Luis Alberto Torres López."
)

