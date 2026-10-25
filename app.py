import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (prevents any truncation or ellipsis)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    * {
        text-overflow: clip !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "airbnb_model.joblib")

@st.cache_resource
def load_airbnb_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

payload = load_airbnb_model()

# ---------------------------------------------------------
# Defaults & Borough Benchmarks
# ---------------------------------------------------------
DEFAULT_BOROUGH_AVGS = {
    'Manhattan': 181.5,
    'Brooklyn': 119.8,
    'Queens': 94.2,
    'Staten Island': 89.2,
    'Bronx': 82.8
}

DEFAULT_BOROUGH_COORDS = {
    'Manhattan': {'lat': 40.7831, 'lon': -73.9712},
    'Brooklyn': {'lat': 40.6782, 'lon': -73.9442},
    'Queens': {'lat': 40.7282, 'lon': -73.7949},
    'Bronx': {'lat': 40.8448, 'lon': -73.8648},
    'Staten Island': {'lat': 40.5795, 'lon': -74.1502}
}

# ---------------------------------------------------------
# Sidebar - Presets & Project Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=70)
    st.title("Project Overview")
    st.markdown("""
    **Pure Machine Learning Application**
    - **Dataset**: Kaggle NYC Airbnb Open Data 2019
    - **Pipeline**: Scikit-Learn `ColumnTransformer` + Gradient Boosting
    - **Techniques**: Outlier filtering ($10 - $800), One-Hot Encoding, StandardScaler
    """)
    st.divider()

    st.subheader("⚡ Quick Listing Presets")
    preset = st.radio(
        "Auto-populate listing profile:",
        options=[
            "Custom Input",
            "🗽 Manhattan Luxury Loft (Entire Apt)",
            "🌿 Brooklyn Cozy Private Room",
            "✈️ Queens Transit Room",
            "🏙️ Bronx Budget Apartment"
        ],
        index=0
    )

if "Manhattan" in preset:
    default_borough = "Manhattan"
    default_room = "Entire home/apt"
    default_min_nights = 2
    default_reviews = 45
    default_rpm = 2.5
    default_host_listings = 1
    default_availability = 280
elif "Brooklyn" in preset:
    default_borough = "Brooklyn"
    default_room = "Private room"
    default_min_nights = 1
    default_reviews = 30
    default_rpm = 1.8
    default_host_listings = 1
    default_availability = 180
elif "Queens" in preset:
    default_borough = "Queens"
    default_room = "Private room"
    default_min_nights = 1
    default_reviews = 15
    default_rpm = 1.0
    default_host_listings = 2
    default_availability = 320
elif "Bronx" in preset:
    default_borough = "Bronx"
    default_room = "Entire home/apt"
    default_min_nights = 3
    default_reviews = 10
    default_rpm = 0.8
    default_host_listings = 1
    default_availability = 120
else:
    default_borough = "Brooklyn"
    default_room = "Entire home/apt"
    default_min_nights = 2
    default_reviews = 20
    default_rpm = 1.5
    default_host_listings = 1
    default_availability = 200

# ---------------------------------------------------------
# Main Page
# ---------------------------------------------------------
st.markdown('<div class="main-header">🗽 NYC Airbnb Nightly Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Accurate, machine-learning powered price valuation for New York City property hosts.</div>', unsafe_allow_html=True)

if payload is None:
    st.error("Model file `airbnb_model.joblib` not found. Please run the notebook `airbnb_price_prediction.ipynb` first to train and export the pipeline.")
    st.stop()

pipeline = payload.get('pipeline', payload)
borough_averages = payload.get('borough_averages', DEFAULT_BOROUGH_AVGS)
borough_coords = payload.get('borough_coords', DEFAULT_BOROUGH_COORDS)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Listing Characteristics")

    with st.expander("📍 Location & Property Type", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            borough = st.selectbox(
                "Borough",
                ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
                index=["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"].index(default_borough)
            )
        with c2:
            room_type = st.selectbox(
                "Room Type",
                ["Entire home/apt", "Private room", "Shared room"],
                index=["Entire home/apt", "Private room", "Shared room"].index(default_room)
            )

    with st.expander("🛏️ Stay & Reservation Rules", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            min_nights = st.slider("Minimum Nights Required", min_value=1, max_value=30, value=default_min_nights, step=1)
            availability = st.slider("Availability (Days per Year)", min_value=0, max_value=365, value=default_availability, step=5)
        with c4:
            reviews_count = st.slider("Total Number of Reviews", min_value=0, max_value=300, value=default_reviews, step=5)
            reviews_per_month = st.slider("Reviews per Month", min_value=0.0, max_value=15.0, value=float(default_rpm), step=0.1)

    host_listings = st.number_input("Number of Listings Managed by Host", min_value=1, max_value=50, value=default_host_listings, step=1)

    eval_btn = st.button("🏷️ Calculate Optimal Nightly Price", type="primary", use_container_width=True)

with col_output:
    st.subheader("📊 Price Valuation & Revenue Estimate")

    # Get borough coordinates
    coords = borough_coords.get(borough, {'lat': 40.7128, 'lon': -74.0060})
    lat = coords['lat']
    lon = coords['lon']

    # Prepare input dataframe
    input_df = pd.DataFrame([{
        'neighbourhood_group': borough,
        'room_type': room_type,
        'latitude': lat,
        'longitude': lon,
        'minimum_nights': min_nights,
        'number_of_reviews': reviews_count,
        'reviews_per_month': reviews_per_month,
        'calculated_host_listings_count': host_listings,
        'availability_365': availability
    }])

    # Predict price
    raw_pred = pipeline.predict(input_df)[0]
    predicted_nightly = max(10.0, round(float(raw_pred), 0))
    monthly_estimated_gross = predicted_nightly * 30 * 0.70  # Assuming 70% occupancy
    borough_benchmark = round(float(borough_averages.get(borough, 120.0)), 0)

    # Clean KPI Cards without grey text or abbreviations
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Recommended Price
            </div>
            <div style="color: #0F172A; font-size: 1.6rem; font-weight: 700; white-space: nowrap;">
                ${predicted_nightly:,.0f}
            </div>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Monthly Revenue
            </div>
            <div style="color: #0F172A; font-size: 1.6rem; font-weight: 700; white-space: nowrap;">
                ${monthly_estimated_gross:,.0f}
            </div>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Borough Average
            </div>
            <div style="color: #0F172A; font-size: 1.6rem; font-weight: 700; white-space: nowrap;">
                ${borough_benchmark:,.0f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Highlight Card
    st.markdown(f"""
    <div style="background-color: #ECFDF5; border: 2px solid #10B981; border-radius: 12px; padding: 22px; text-align: center; margin-bottom: 20px;">
        <span style="color: #047857; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Suggested Listing Price</span>
        <h1 style="color: #065F46; font-size: 2.5rem; margin: 6px 0;">${predicted_nightly:,.0f} <span style="font-size: 1.1rem; color: #059669; font-weight: 400;">/ night</span></h1>
        <p style="color: #047857; font-size: 0.95rem; margin: 0;">
            Estimated gross monthly revenue of <b>${monthly_estimated_gross:,.0f}</b> based on typical 70% seasonal occupancy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Market Comparison Analysis
    st.markdown("### 💡 Pricing Insights & Market Context")
    diff = predicted_nightly - borough_benchmark
    diff_pct = (diff / borough_benchmark) * 100

    if diff > 15:
        st.markdown(f"- 📈 **Premium Positioning**: Your listing is priced **{diff_pct:.0f}% higher** than the {borough} borough benchmark (\\${borough_benchmark:,.0f}/night), reflecting its high-demand room type or location appeal.")
    elif diff < -15:
        st.markdown(f"- 🏷️ **Competitive Advantage**: Your listing is priced **{abs(diff_pct):.0f}% below** the {borough} borough average (\\${borough_benchmark:,.0f}/night), optimizing for rapid occupancy and booking velocity.")
    else:
        st.markdown(f"- ⚖️ **Market Aligned**: Your recommended price aligns closely with average listings in {borough} (\\${borough_benchmark:,.0f}/night).")

    if room_type == "Entire home/apt":
        st.markdown("- 🏠 **Entire Home Premium**: Entire apartments in NYC consistently command higher average pricing compared to private rooms.")
    elif room_type == "Private room":
        st.markdown("- 🚪 **Private Room Economy**: Private rooms offer competitive pricing that appeals to budget solo travelers and digital nomads.")
    else:
        st.markdown("- 👥 **Shared Space**: Shared accommodations compete primarily on affordability and lower nightly rates.")
