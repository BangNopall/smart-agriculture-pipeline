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

# Database connection
@st.cache_resource
def get_db_connection():
    """Create database connection"""
    engine = create_engine("postgresql://postgres:password@localhost:5432/agriculture")
    return engine

# Load data
@st.cache_data
def load_data():
    """Load data from database"""
    engine = get_db_connection()
    
    # Only load integrated data (one-year data)
    query = "SELECT * FROM integrated_agriculture_data"
    return pd.read_sql(query, engine)

# Load ML model
@st.cache_resource
def load_model():
    """Load trained ML model"""
    model_path = Path("models/harvest_predictor_v1.pkl")
    if model_path.exists():
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['feature_columns']
    return None, None

# ========================================
# MAIN APP
# ========================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Smart Agriculture Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Prediksi dan Visualisasi Produktivitas Panen Padi (2024)")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/2E7D32/FFFFFF?text=Smart+Agriculture", width='stretch')
        st.markdown("---")
        
        page = st.radio(
            "Navigasi",
            ["📊 Overview", "🗺️ Regional Analysis", "🌦️ Weather Impact", "🤖 Prediction Tool", "📈 Insights"]
        )
        
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("- BPS Indonesia")
        st.markdown("- BMKG Weather API")
        st.markdown("- FAO Statistics")
    
    # Load data
    try:
        df = load_data()
        df['provinsi'] = df['provinsi'].str.title()  # Capitalize province names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Route to different pages
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

# ========================================
# PAGE FUNCTIONS
# ========================================

def show_overview(df):
    """Overview page with key metrics"""
    st.header("📊 Overview Dashboard (2024)")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_productivity = df['produktivitas_ton_per_ha'].mean()
        st.metric("Avg Productivity", f"{avg_productivity:.2f} ton/ha")
    
    with col2:
        total_production = df['produksi_ton'].sum() / 1000000
        st.metric("Total Production", f"{total_production:.1f}M ton")
    
    with col3:
        total_area = df['luas_panen_ha'].sum() / 1000000
        st.metric("Total Area", f"{total_area:.1f}M ha")
    
    with col4:
        regions_count = df['provinsi'].nunique()
        st.metric("Regions Covered", regions_count)
    
    st.markdown("---")
    
    # Productivity distribution
    st.subheader("Productivity Distribution Across Indonesia")
    fig = px.box(df, y='produktivitas_ton_per_ha', title="Productivity Distribution Across Regions")
    fig.update_traces(marker_color='#4CAF50')
    st.plotly_chart(fig, width='stretch')
    
    # Top performing regions
    st.subheader("🏆 Top 10 Performing Regions")
    top_regions = df.nlargest(10, 'produktivitas_ton_per_ha')[
        ['provinsi', 'produktivitas_ton_per_ha', 'produksi_ton', 'suhu_rata_c', 'curah_hujan_mm']
    ]
    st.dataframe(top_regions, width='stretch')

def show_regional_analysis(df):
    """Regional analysis page"""
    st.header("🗺️ Regional Analysis")
    
    # Region selector
    regions = sorted(df['provinsi'].unique())
    selected_region = st.selectbox("Pilih Provinsi", regions)
    
    df_region = df[df['provinsi'] == selected_region]
    
    # Region metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prod = df_region['produktivitas_ton_per_ha'].iloc[0]
        st.metric("Productivity", f"{prod:.2f} ton/ha")
    
    with col2:
        temp = df_region['suhu_rata_c'].iloc[0]
        st.metric("Avg Temperature", f"{temp:.1f}°C")
    
    with col3:
        rain = df_region['curah_hujan_mm'].iloc[0]
        st.metric("Rainfall", f"{rain:.0f} mm")
    
    # Show full details
    st.subheader(f"Detail Data - {selected_region}")
    st.dataframe(df_region.T.rename(columns={df_region.index[0]: selected_region}))

def show_weather_impact(df):
    """Weather impact analysis"""
    st.header("🌦️ Weather Impact on Productivity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature vs Productivity")
        fig = px.scatter(df, x='suhu_rata_c', y='produktivitas_ton_per_ha',
                         size='produksi_ton', color='provinsi',
                         labels={'suhu_rata_c': 'Temperature (°C)'})
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Rainfall vs Productivity")
        fig = px.scatter(df, x='curah_hujan_mm', y='produktivitas_ton_per_ha',
                         size='produksi_ton', color='provinsi',
                         labels={'curah_hujan_mm': 'Rainfall (mm)'})
        st.plotly_chart(fig, width='stretch')

def show_prediction_tool(df):
    """Interactive prediction tool"""
    st.header("🤖 Harvest Prediction Tool")
    
    st.info("Sesuaikan parameter di bawah ini untuk memprediksi produktivitas panen.")
    
    # Load model
    model, feature_cols = load_model()
    
    if model is None:
        st.warning("Model belum tersedia. Harap latih model terlebih dahulu.")
        return
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        suhu = st.slider("Temperature (°C)", 20.0, 35.0, 26.5, 0.5)
        curah_hujan = st.slider("Rainfall (mm)", 1000, 4000, 2000, 100)
        kelembaban = st.slider("Humidity (%)", 60, 95, 75, 5)
        hari_hujan = st.slider("Rainy Days", 80, 180, 120, 10)
    
    with col2:
        luas_panen = st.number_input("Harvest Area (ha)", 10000, 100000, 50000, 5000)
        drought_index = curah_hujan / (suhu * 100)
        humidity_stress = 1 if kelembaban < 60 or kelembaban > 90 else 0
        interaction = curah_hujan * suhu
    
    # Predict
    if st.button("🎯 Predict Productivity", type="primary"):
        input_data = pd.DataFrame({
            'suhu_rata_c': [suhu],
            'curah_hujan_mm': [curah_hujan],
            'kelembaban_persen': [kelembaban],
            'hari_hujan': [hari_hujan],
            'luas_panen_ha': [luas_panen],
            'drought_index': [drought_index],
            'humidity_stress': [humidity_stress],
            'curah_hujan_x_suhu': [interaction]
        })
        
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Productivity: {prediction:.2f} ton/ha")

def show_insights(df):
    """Insights and analysis"""
    st.header("📈 Insights")
    
    # Top and bottom regions
    st.subheader("Best and Worst Performing Regions")
    top_region = df.nlargest(1, 'produktivitas_ton_per_ha')
    bottom_region = df.nsmallest(1, 'produktivitas_ton_per_ha')
    
    st.write("🏆 **Best:**")
    st.dataframe(top_region[['provinsi', 'produktivitas_ton_per_ha']])
    
    st.write("❌ **Worst:**")
    st.dataframe(bottom_region[['provinsi', 'produktivitas_ton_per_ha']])
    
    # Correlation heatmap
    st.subheader("Feature Correlation")
    corr = df[['suhu_rata_c', 'curah_hujan_mm', 'kelembaban_persen', 'hari_hujan', 'produktivitas_ton_per_ha']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, width='stretch')

# Run app
if __name__ == "__main__":
    main()
