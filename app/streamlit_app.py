from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="VerdantAI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "crop_yield_dataset.csv"
MODEL_PATH = BASE_DIR / "notebooks" / "model.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=["Crop_Yield", "Date", "Soil_Quality"])
X = pd.get_dummies(X, drop_first=True)


# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top right, #12351f 0%, #070d0a 45%, #050706 100%);
}

.block-container {
    padding-top: 3rem;
    max-width: 1200px;
}

.botanical-card {
    background:
        radial-gradient(circle at top right, rgba(139, 207, 99, 0.12), transparent 35%),
        linear-gradient(135deg, rgba(20, 55, 32, 0.92), rgba(7, 15, 10, 0.95));
    border: 1px solid rgba(126, 200, 120, 0.35);
    border-radius: 24px;
    padding: 1.8rem;
    margin: 1.4rem 0;
    box-shadow: 0 0 35px rgba(90, 200, 110, 0.12);
}

.botanical-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #E8F5E9;
    margin-bottom: 0.35rem;
}

.botanical-subtitle {
    color: #A8C686;
    font-size: 1rem;
}

.metric-card {
    background:
        radial-gradient(circle at top right, rgba(139, 207, 99, 0.18), transparent 35%),
        linear-gradient(135deg, rgba(16, 58, 34, 0.95), rgba(7, 16, 11, 0.98));
    border: 1px solid rgba(139, 207, 99, 0.38);
    border-radius: 24px;
    padding: 1.7rem;
    min-height: 175px;
    box-shadow: 0 0 35px rgba(90, 200, 110, 0.12);
}

.metric-card h3 {
    color: #A8E07A;
    font-size: 1.05rem;
    margin-bottom: 0.8rem;
}

.metric-card h1 {
    color: #F4F1E8;
    font-size: 2.4rem;
    margin: 0;
}

.metric-card p {
    color: #BFD0B8;
    font-size: 0.92rem;
    margin-top: 0.8rem;
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    color: #F4F1E8;
    margin-bottom: 0;
}

.hero-title span {
    color: #8BCF63;
}

.hero-subtitle {
    color: #9DD37B;
    font-size: 1.25rem;
    margin-top: 0.4rem;
}

.soft-text {
    color: #C9D8C2;
    font-size: 1rem;
}

.leaf-divider {
    text-align: center;
    color: #8BCF63;
    margin: 2rem 0;
    letter-spacing: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    logo_path = BASE_DIR / "visualisations" / "logo.png"
    if logo_path.exists():
        st.image(logo_path, width=180)

    st.markdown("### 🌱 VerdantAI")
    with st.expander("About"):
        st.markdown("""
        **Model:** Random Forest Regressor  
        **Version:** v1.0  
        **Focus:** Sustainable Agriculture  
        """)


# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero-title">Verdant<span>AI</span> 🌱</div>
<div class="hero-subtitle">Environmental Intelligence for Crop Yield Prediction</div>
<p class="soft-text">
VerdantAI explores how climate, soil chemistry, and environmental conditions influence agricultural productivity.
</p>
""", unsafe_allow_html=True)

st.divider()


# -----------------------------
# Environmental controls
# -----------------------------
st.markdown("""
<div class="botanical-card">
    <div class="botanical-title">🌿 Environmental Controls</div>
    <div class="botanical-subtitle">
        Adjust crop, soil, nutrient and climate parameters to simulate predicted yield response.
    </div>
</div>
""", unsafe_allow_html=True)

crop_type = st.selectbox(
    "Crop Type",
    ["Corn", "Cotton", "Potato", "Rice", "Soybean", "Sugarcane", "Sunflower", "Tomato", "Wheat"],
)

soil_type = st.selectbox(
    "Soil Type",
    ["Loamy", "Peaty", "Saline", "Sandy"],
)

left, right = st.columns(2)

with left:
    temperature = st.slider("Temperature (°C)", 0.0, 45.0, 20.0)
    soil_ph = st.slider("Soil pH", 3.0, 9.0, 6.5)
    phosphorus = st.slider("Phosphorus (P)", 0.0, 100.0, 40.0)

with right:
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
    nitrogen = st.slider("Nitrogen (N)", 0.0, 100.0, 50.0)
    potassium = st.slider("Potassium (K)", 0.0, 100.0, 40.0)


# -----------------------------
# Prediction input
# -----------------------------
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

soil_column = f"Soil_Type_{soil_type}"
if soil_column in input_data.columns:
    input_data.loc[0, soil_column] = 1

prediction = model.predict(input_data)[0]


# -----------------------------
# Suitability and resilience
# -----------------------------
risk_score = 100

if temperature > 35:
    risk_score -= 25

if humidity < 30:
    risk_score -= 20

if soil_ph < 5.5:
    risk_score -= 15

if nitrogen < 30:
    risk_score -= 10

resilience_score = (
    (100 - abs(temperature - 24) * 2)
    + humidity
    + nitrogen
) / 3

confidence_label = "High" if risk_score >= 75 else "Moderate" if risk_score >= 50 else "Low"


# -----------------------------
# Dashboard cards
# -----------------------------
st.markdown('<div class="leaf-divider">❦ ❦ ❦</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌾 Predicted Crop Yield</h3>
        <h1>{prediction:.2f} t/ha</h1>
        <p>Estimated yield from selected crop, soil and climate conditions.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌍 Environmental Suitability</h3>
        <h1>{risk_score}/100</h1>
        <p>Stress-adjusted score based on heat, humidity, pH and nitrogen.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🧠 Model Confidence</h3>
        <h1>{confidence_label}</h1>
        <p>Random Forest prediction based on environmental input patterns.</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Environmental interpretation
# -----------------------------
st.markdown("""
<div class="botanical-card">
    <div class="botanical-title">🌍 Environmental Assessment</div>
    <div class="botanical-subtitle">
        Real-time interpretation of crop stress and environmental resilience.
    </div>
</div>
""", unsafe_allow_html=True)

stress_detected = False

if temperature > 35:
    st.warning("⚠️ High temperature stress detected. Irrigation and heat-resistant cultivars may be beneficial.")
    stress_detected = True

if humidity < 30:
    st.warning("⚠️ Low humidity detected — drought stress conditions are possible.")
    stress_detected = True

if soil_ph < 5.5:
    st.warning("⚠️ Soil acidity may reduce nutrient availability and root uptake.")
    stress_detected = True

if not stress_detected:
    st.success(f"🌱 Current conditions are favourable for {crop_type} growth.")

if prediction > 40:
    st.success("🌱 Exceptional productivity conditions detected.")
elif prediction > 25:
    st.info("🌿 Conditions are moderately favourable for sustainable crop growth.")
else:
    st.warning("⚠️ Environmental stressors may significantly reduce agricultural productivity.")

st.metric("Resilience Score", f"{resilience_score:.1f}/100")


# -----------------------------
# Model insights
# -----------------------------
st.markdown('<div class="leaf-divider">❦ ❦ ❦</div>', unsafe_allow_html=True)

st.markdown("""
<div class="botanical-card">
    <div class="botanical-title">🧠 Model Insights</div>
    <div class="botanical-subtitle">
        Understanding the environmental drivers behind the Random Forest model.
    </div>
</div>
""", unsafe_allow_html=True)

feature_image = BASE_DIR / "visualisations" / "feature_importance.png"
if feature_image.exists():
    st.image(feature_image, caption="Random Forest Feature Importance Analysis", width="stretch")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

fig_importance = px.bar(
    feature_importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Greens",
    title="Top Environmental Drivers",
)

fig_importance.update_layout(
    paper_bgcolor="#0B1020",
    plot_bgcolor="#0B1020",
    font=dict(color="white", size=14),
    title_font=dict(size=24),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
    yaxis=dict(showgrid=False),
)

st.plotly_chart(fig_importance, width="stretch")


# -----------------------------
# Environmental visualisations
# -----------------------------
st.markdown('<div class="leaf-divider">❦ ❦ ❦</div>', unsafe_allow_html=True)

st.markdown("""
<div class="botanical-card">
    <div class="botanical-title">📊 Environmental Visualisations</div>
    <div class="botanical-subtitle">
        Explore how temperature, humidity and crop type relate to predicted agricultural productivity.
    </div>
</div>
""", unsafe_allow_html=True)

fig_temp = px.scatter(
    df,
    x="Temperature",
    y="Crop_Yield",
    color="Crop_Type",
    hover_data=["Soil_Type", "Humidity", "N", "P", "K"],
    title="Crop Yield Response to Temperature",
    labels={
        "Temperature": "Temperature (°C)",
        "Crop_Yield": "Crop Yield (t/ha)",
        "Crop_Type": "Crop Type",
    },
    template="plotly_dark",
)

fig_temp.update_traces(marker=dict(size=7, opacity=0.65, line=dict(width=0.5, color="#E8F5E9")))
fig_temp.update_layout(
    title_font_size=24,
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="#E8F5E9"),
)

st.plotly_chart(fig_temp, width="stretch")

fig_humidity = px.scatter(
    df,
    x="Humidity",
    y="Crop_Yield",
    color="Crop_Type",
    hover_data=["Soil_Type", "Temperature", "N", "P", "K"],
    title="Crop Yield Response to Humidity",
    labels={
        "Humidity": "Humidity (%)",
        "Crop_Yield": "Crop Yield (t/ha)",
        "Crop_Type": "Crop Type",
    },
    template="plotly_dark",
)

fig_humidity.update_traces(marker=dict(size=7, opacity=0.65, line=dict(width=0.5, color="#E8F5E9")))
fig_humidity.update_layout(
    title_font_size=24,
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="#E8F5E9"),
)

st.plotly_chart(fig_humidity, width="stretch")


# -----------------------------
# Botanical infographic
# -----------------------------
st.markdown('<div class="leaf-divider">❦ ❦ ❦</div>', unsafe_allow_html=True)

st.markdown("""
<div class="botanical-card">
    <div class="botanical-title">🌿 Crop Yield Distribution</div>
    <div class="botanical-subtitle">
        Botanical summary of crop yield ranges and distribution patterns.
    </div>
</div>
""", unsafe_allow_html=True)

distribution_image = BASE_DIR / "visualisations" / "crop_yield_distribution.png"
if distribution_image.exists():
    st.image(distribution_image, caption="Crop yield distribution across plant types", width="stretch")