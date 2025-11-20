import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Smart Agriculture Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATABASE & MODEL LOADING
# =========================================================

@st.cache_resource
def get_db_connection():
    """Create database connection"""
    # TODO: sesuaikan dengan konfigurasi DB-mu (user, password, host, dbname)
    engine = create_engine("postgresql://postgres:password@localhost:5432/agriculture")
    return engine

@st.cache_data
def load_data():
    """Load integrated data from database (2024 only)"""
    engine = get_db_connection()
    query = "SELECT * FROM integrated_agriculture_data"
    df = pd.read_sql(query, engine)
    return df

@st.cache_resource
def load_model():
    """Load trained ML model"""
    model_path = Path("models/harvest_predictor_v1.pkl")
    if model_path.exists():
        model_data = joblib.load(model_path)
        return model_data["model"], model_data["feature_columns"]
    return None, None

# =========================================================
# MAIN APP
# =========================================================

def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🌾 Smart Agriculture Dashboard</h1>',
        unsafe_allow_html=True
    )
    st.markdown("### Prediksi dan Visualisasi Produktivitas Panen Padi (Tahun 2024)")
    
    # Sidebar
    with st.sidebar:
        st.image(
            "https://placehold.co/300x100/000000/FFF?text=Smart+Agriculture",
            width='stretch'
        )
        st.markdown("---")
        
        page = st.radio(
            "Navigasi",
            ["📊 Overview", "🗺️ Regional Analysis", "🌦️ Weather Impact", "🤖 Prediction Tool", "📈 Insights"]
        )
        
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("- BPS Indonesia (Produksi Padi 2024)")
        st.markdown("- BMKG (Data Iklim Rata-rata 2024)")
    
    # Load data
    try:
        df = load_data()
        # Standarisasi display nama provinsi
        df["provinsi"] = df["provinsi"].str.title()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Routing
    if page == "📊 Overview":
        show_overview(df)
    elif page == "🗺️ Regional Analysis":
        show_regional_analysis(df)
    elif page == "🌦️ Weather Impact":
        show_weather_impact(df)
    elif page == "🤖 Prediction Tool":
        show_prediction_tool(df)
    elif page == "📈 Insights":
        show_insights(df)

# =========================================================
# PAGE FUNCTIONS
# =========================================================

def show_overview(df: pd.DataFrame):
    """Overview page with key metrics"""
    st.header("📊 Overview Dashboard (2024)")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_productivity = df["produktivitas_ton_per_ha"].mean()
        st.metric("Avg Productivity", f"{avg_productivity:.2f} ton/ha")
    
    with col2:
        total_production = df["produksi_ton"].sum() / 1_000_000
        st.metric("Total Production", f"{total_production:.1f}M ton")
    
    with col3:
        total_area = df["luas_panen_ha"].sum() / 1_000_000
        st.metric("Total Area", f"{total_area:.1f}M ha")
    
    with col4:
        regions_count = df["provinsi"].nunique()
        st.metric("Regions Covered", regions_count)
    
    st.markdown("---")
    
    # Productivity distribution
    st.subheader("Productivity Distribution Across Indonesia")
    fig_box = px.box(
        df,
        y="produktivitas_ton_per_ha",
        title="Productivity Distribution Across Regions"
    )
    st.plotly_chart(fig_box, width='stretch')
    
    # Top performing regions
    st.subheader("🏆 Top 10 Performing Regions")
    top_regions = df.nlargest(
        10, "produktivitas_ton_per_ha"
    )[
        ["provinsi", "produktivitas_ton_per_ha", "produksi_ton", "suhu_rata_c", "curah_hujan_mm"]
    ]
    st.dataframe(top_regions, width='stretch')

def show_regional_analysis(df: pd.DataFrame):
    """Regional analysis page"""
    st.header("🗺️ Regional Analysis")
    
    regions = sorted(df["provinsi"].unique())
    selected_region = st.selectbox("Pilih Provinsi", regions)
    
    df_region = df[df["provinsi"] == selected_region]
    
    if df_region.empty:
        st.warning("Data tidak ditemukan untuk provinsi yang dipilih.")
        return
    
    # Region metrics
    col1, col2, col3 = st.columns(3)
    row = df_region.iloc[0]
    
    with col1:
        st.metric("Productivity", f"{row['produktivitas_ton_per_ha']:.2f} ton/ha")
    with col2:
        st.metric("Avg Temperature", f"{row['suhu_rata_c']:.1f} °C")
    with col3:
        st.metric("Rainfall", f"{row['curah_hujan_mm']:.0f} mm")
    
    # Detail data
    st.subheader(f"Detail Data - {selected_region}")
    st.dataframe(df_region.T, width='stretch')

def show_weather_impact(df: pd.DataFrame):
    """Weather impact analysis"""
    st.header("🌦️ Weather Impact on Productivity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature vs Productivity")
        fig_temp = px.scatter(
            df,
            x="suhu_rata_c",
            y="produktivitas_ton_per_ha",
            size="produksi_ton",
            color="provinsi",
            labels={"suhu_rata_c": "Temperature (°C)", "produktivitas_ton_per_ha": "Productivity (ton/ha)"},
            hover_name="provinsi"
        )
        st.plotly_chart(fig_temp, width='stretch')
    
    with col2:
        st.subheader("Rainfall vs Productivity")
        fig_rain = px.scatter(
            df,
            x="curah_hujan_mm",
            y="produktivitas_ton_per_ha",
            size="produksi_ton",
            color="provinsi",
            labels={"curah_hujan_mm": "Rainfall (mm)", "produktivitas_ton_per_ha": "Productivity (ton/ha)"},
            hover_name="provinsi"
        )
        st.plotly_chart(fig_rain, width='stretch')

def show_prediction_tool(df: pd.DataFrame):
    """Interactive prediction tool (sesuai feature set model terbaru)"""
    st.header("🤖 Harvest Prediction Tool")
    st.info("Sesuaikan parameter di bawah ini untuk memprediksi produktivitas panen (ton/ha).")
    
    # Load model
    model, feature_cols = load_model()
    
    if model is None or feature_cols is None:
        st.warning("Model belum tersedia. Harap latih model terlebih dahulu.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        suhu = st.slider("Temperature (°C)", 20.0, 35.0, 26.5, 0.5)
        curah_hujan = st.slider("Annual Rainfall (mm)", 1000, 4000, 2000, 100)
        kelembaban = st.slider("Average Humidity (%)", 60, 95, 80, 1)
        hari_hujan = st.slider("Rainy Days (per year)", 60, 200, 120, 5)
    
    with col2:
        luas_panen = st.number_input("Harvest Area (ha)", 1_000, 200_000, 50_000, 1_000)
        # Fitur turunan sesuai pipeline transform.py
        drought_index = curah_hujan / (suhu * 100)
        interaction = curah_hujan * suhu
        
        st.caption(f"📌 Drought index (approx): {drought_index:.4f}")
    
    # Predict
    if st.button("🎯 Predict Productivity", type="primary"):
        # DataFrame input harus punya kolom yang sama dengan feature_columns model
        raw_input = {
            "suhu_rata_c": [suhu],
            "curah_hujan_mm": [curah_hujan],
            "kelembaban_persen": [kelembaban],
            "hari_hujan": [hari_hujan],
            "luas_panen_ha": [luas_panen],
            "drought_index": [drought_index],
            "curah_hujan_x_suhu": [interaction],
        }
        
        input_df = pd.DataFrame(raw_input)
        
        # Reorder kolom sesuai feature_cols dari model_training
        try:
            input_df = input_df[feature_cols]
        except KeyError as e:
            st.error(f"Kolom input tidak cocok dengan model: {e}")
            st.write("Fitur yang diminta model:", feature_cols)
            st.write("Fitur yang tersedia di input:", list(input_df.columns))
            return
        
        prediction = model.predict(input_df)[0]
        st.success(f"### Predicted Productivity: **{prediction:.2f} ton/ha**")

def show_insights(df: pd.DataFrame):
    """Insights and analysis"""
    st.header("📈 Insights")
    
    # Best and worst regions
    st.subheader("Best and Worst Performing Regions (2024)")
    top_region = df.nlargest(1, "produktivitas_ton_per_ha")
    bottom_region = df.nsmallest(1, "produktivitas_ton_per_ha")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("🏆 **Best Region:**")
        st.dataframe(top_region[["provinsi", "produktivitas_ton_per_ha"]], width='stretch')
    with col2:
        st.write("❌ **Lowest Region:**")
        st.dataframe(bottom_region[["provinsi", "produktivitas_ton_per_ha"]], width='stretch')
    
    st.markdown("---")
    
    # Correlation heatmap for main numeric variables
    st.subheader("Feature Correlation (2024)")
    numeric_cols = [
        "suhu_rata_c",
        "curah_hujan_mm",
        "kelembaban_persen",
        "hari_hujan",
        "luas_panen_ha",
        "produktivitas_ton_per_ha",
    ]
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(available_cols) >= 2:
        corr = df[available_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, width='stretch')
    else:
        st.info("Kolom numerik untuk korelasi tidak mencukupi.")

# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    main()