from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "crop_yield_dataset.csv"

df = pd.read_csv(data_path)
model = joblib.load(BASE_DIR / "notebooks" / "model.pkl")
import streamlit as st

X = df.drop(columns=["Crop_Yield", "Date", "Soil_Quality"])
X = pd.get_dummies(X, drop_first=True)

st.set_page_config(
    page_title="VerdantAI",
    layout="wide"
)

st.title("🌱 VerdantAI")
st.subheader("Environmental Intelligence for Crop Yield Prediction")




st.markdown("""
VerdantAI explores how climate, soil chemistry,
and environmental conditions influence agricultural productivity.
""")

st.divider()

st.header("🌿 Environmental Controls")

crop_type = st.selectbox(
    "Crop Type",
    [
        "Corn",
        "Cotton",
        "Potato",
        "Rice",
        "Soybean",
        "Sugarcane",
        "Sunflower",
        "Tomato",
        "Wheat"
    ]
)

temperature = st.slider(
    "Temperature (°C)",
    min_value=0.0,
    max_value=45.0,
    value=20.0
)

humidity = st.slider(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

soil_ph = st.slider(
    "Soil pH",
    min_value=3.0,
    max_value=9.0,
    value=6.5
)

nitrogen = st.slider(
    "Nitrogen (N)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

phosphorus = st.slider(
    "Phosphorus (P)",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

potassium = st.slider(
    "Potassium (K)",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

risk_score = 100
st.metric(
    label="🌍 Environmental Suitability",
    value=f"{risk_score}/100"
)
if temperature > 35:
    risk_score -= 25
    

if humidity < 30:
    risk_score -= 20
    

if soil_ph < 5.5:
    risk_score -= 15

if nitrogen < 30:
    risk_score -= 10

st.progress(risk_score / 100)
st.metric("Environmental Suitability", f"{risk_score}/100")

st.divider()

st.subheader("🌱 Current Environmental Conditions")

input_data = pd.DataFrame(0.0, index=[0], columns=X.columns)

input_data.loc[0, "Soil_pH"] = float(soil_ph)
input_data.loc[0, "Temperature"] = float(temperature)
input_data.loc[0, "Humidity"] = float(humidity)
input_data.loc[0, "Wind_Speed"] = 5.0
input_data.loc[0, "N"] = float(nitrogen)
input_data.loc[0, "P"] = float(phosphorus)
input_data.loc[0, "K"] = float(potassium)

crop_column = f"Crop_Type_{crop_type}"

if crop_column in input_data.columns:
    input_data.loc[0, crop_column] = 1

soil_type = st.selectbox(
    "Soil Type",
    [
        "Loamy",
        "Peaty",
        "Saline",
        "Sandy"
    ]
)

soil_column = f"Soil_Type_{soil_type}"

if soil_column in input_data.columns:
    input_data.loc[0, soil_column] = 1

prediction = model.predict(input_data)[0]
st.metric(
    label="Predicted Yield",
    value=f"{prediction:.2f} t/ha"
)

if temperature > 35:
    st.warning("⚠️ High temperature stress may reduce crop productivity.")
    st.info("Irrigation recommended to reduce drought stress.")

if humidity < 30:
    st.warning("⚠️ Low humidity detected — drought conditions possible.")
    st.info("Irrigation recommended to reduce drought stress.")

if soil_ph < 5.5:
    st.warning("⚠️ Soil acidity may negatively affect nutrient uptake.")
st.divider()

stress_detected = False

if temperature > 35:
    st.warning("⚠️ High temperature stress detected.")
    stress_detected = True

if humidity < 30:
    st.warning("⚠️ Low humidity may reduce productivity.")
    stress_detected = True

if soil_ph < 5.5:
    st.warning("⚠️ Soil acidity may affect nutrient absorption.")
    stress_detected = True

if not stress_detected:
    st.success(
        f"🌱 Current conditions are favourable for {crop_type} growth."
    )

st.header("🌾 Predicted Crop Yield")
st.divider()

st.header("🧠 Model Insights")

st.image(
    "visualisations/feature_importance.png",
    caption="Random Forest Feature Importance Analysis"
)

st.write({
    "Temperature": temperature,
    "Humidity": humidity,
    "Soil pH": soil_ph,
    "Nitrogen": nitrogen,
    "Phosphorus": phosphorus,
    "Potassium": potassium
})

