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
    
    queries = {
        'integrated': "SELECT * FROM integrated_agriculture_data",
        'productivity_trend': "SELECT * FROM v_productivity_trend",
        'weather_impact': "SELECT * FROM v_weather_impact"
    }
    
    data = {}
    for key, query in queries.items():
        data[key] = pd.read_sql(query, engine)
    
    return data

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
    st.markdown("### Prediksi Produktivitas Panen untuk Zero Hunger")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/2E7D32/FFFFFF?text=Smart+Agriculture", use_container_width=True)
        st.markdown("---")
        
        page = st.radio(
            "Navigasi",
            ["📊 Overview", "🗺️ Regional Analysis", "🌦️ Weather Impact", "🤖 Prediction Tool", "📈 Trends"]
        )
        
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("- BPS Indonesia")
        st.markdown("- BMKG Weather API")
        st.markdown("- FAO Statistics")
    
    # Load data
    try:
        data = load_data()
        df = data['integrated']
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Using sample data for demonstration...")
        df = generate_sample_data()
    
    # Route to different pages
    if page == "📊 Overview":
        show_overview(df)
    elif page == "🗺️ Regional Analysis":
        show_regional_analysis(df)
    elif page == "🌦️ Weather Impact":
        show_weather_impact(df)
    elif page == "🤖 Prediction Tool":
        show_prediction_tool(df)
    elif page == "📈 Trends":
        show_trends(df)

# ========================================
# PAGE FUNCTIONS
# ========================================

def show_overview(df):
    """Overview page with key metrics"""
    st.header("📊 Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_productivity = df['produktivitas_ton_per_ha'].mean()
        st.metric("Avg Productivity", f"{avg_productivity:.2f} ton/ha", "+5.2%")
    
    with col2:
        total_production = df['produksi_ton'].sum() / 1000000
        st.metric("Total Production", f"{total_production:.1f}M ton", "+3.1%")
    
    with col3:
        total_area = df['luas_panen_ha'].sum() / 1000000
        st.metric("Total Area", f"{total_area:.1f}M ha", "-1.2%")
    
    with col4:
        regions_count = df['provinsi'].nunique()
        st.metric("Regions Covered", regions_count, "100%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Productivity by Year")
        yearly = df.groupby('tahun')['produktivitas_ton_per_ha'].mean().reset_index()
        fig = px.line(yearly, x='tahun', y='produktivitas_ton_per_ha',
                     markers=True, title="Average Productivity Trend")
        fig.update_traces(line_color='#2E7D32', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Production Distribution")
        fig = px.box(df, y='produktivitas_ton_per_ha', 
                    title="Productivity Distribution Across Regions")
        fig.update_traces(marker_color='#4CAF50')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performing regions
    st.subheader("🏆 Top 10 Performing Regions (2022)")
    df_2022 = df[df['tahun'] == df['tahun'].max()]
    top_regions = df_2022.nlargest(10, 'produktivitas_ton_per_ha')[
        ['region_key', 'produktivitas_ton_per_ha', 'produksi_ton', 'suhu_rata_c', 'curah_hujan_mm']
    ]
    st.dataframe(top_regions, use_container_width=True)

def show_regional_analysis(df):
    """Regional analysis page"""
    st.header("🗺️ Regional Analysis")
    
    # Region selector
    regions = sorted(df['provinsi'].unique())
    selected_region = st.selectbox("Select Provinsi", regions)
    
    df_region = df[df['provinsi'] == selected_region]
    
    # Region metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        latest_prod = df_region[df_region['tahun'] == df_region['tahun'].max()]['produktivitas_ton_per_ha'].values[0]
        st.metric("Latest Productivity", f"{latest_prod:.2f} ton/ha")
    
    with col2:
        avg_temp = df_region['suhu_rata_c'].mean()
        st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    with col3:
        avg_rainfall = df_region['curah_hujan_mm'].mean()
        st.metric("Avg Rainfall", f"{avg_rainfall:.0f} mm")
    
    # Time series chart
    st.subheader(f"Productivity Trend - {selected_region}")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_region['tahun'],
        y=df_region['produktivitas_ton_per_ha'],
        mode='lines+markers',
        name='Productivity',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Productivity (ton/ha)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Historical Data")
    st.dataframe(
        df_region[['tahun', 'produktivitas_ton_per_ha', 'suhu_rata_c', 
                   'curah_hujan_mm', 'kelembaban_persen']].sort_values('tahun', ascending=False),
        use_container_width=True
    )

def show_weather_impact(df):
    """Weather impact analysis"""
    st.header("🌦️ Weather Impact on Productivity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Impact")
        fig = px.scatter(df, x='suhu_rata_c', y='produktivitas_ton_per_ha',
                        color='tahun', size='produksi_ton',
                        title="Temperature vs Productivity",
                        labels={'suhu_rata_c': 'Temperature (°C)', 
                               'produktivitas_ton_per_ha': 'Productivity (ton/ha)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Rainfall Impact")
        fig = px.scatter(df, x='curah_hujan_mm', y='produktivitas_ton_per_ha',
                        color='tahun', size='produksi_ton',
                        title="Rainfall vs Productivity",
                        labels={'curah_hujan_mm': 'Rainfall (mm)', 
                               'produktivitas_ton_per_ha': 'Productivity (ton/ha)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Weather-Productivity Correlation")
    weather_cols = ['suhu_rata_c', 'curah_hujan_mm', 'kelembaban_persen', 
                    'hari_hujan', 'produktivitas_ton_per_ha']
    corr_matrix = df[weather_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdYlGn',
                    title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_tool(df):
    """Interactive prediction tool"""
    st.header("🤖 Harvest Prediction Tool")
    
    st.info("💡 Adjust the parameters below to predict productivity")
    
    # Load model
    model, feature_cols = load_model()
    
    if model is None:
        st.warning("Model not found. Please train the model first.")
        return
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weather Parameters")
        suhu = st.slider("Temperature (°C)", 20.0, 35.0, 26.5, 0.5)
        curah_hujan = st.slider("Rainfall (mm)", 1000, 4000, 2000, 100)
        kelembaban = st.slider("Humidity (%)", 60, 95, 75, 5)
        hari_hujan = st.slider("Rainy Days", 80, 180, 120, 10)
    
    with col2:
        st.subheader("Agricultural Parameters")
        luas_panen = st.number_input("Harvest Area (ha)", 10000, 100000, 50000, 5000)
        
        # Calculated features
        drought_index = curah_hujan / (suhu * 100)
        humidity_stress = 1 if kelembaban < 60 or kelembaban > 90 else 0
        interaction = curah_hujan * suhu
    
    # Make prediction
    if st.button("🎯 Predict Productivity", type="primary"):
        # Prepare input
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
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.success(f"### Predicted Productivity: {prediction:.2f} ton/ha")
        
        # Compare with average
        avg_productivity = df['produktivitas_ton_per_ha'].mean()
        diff = ((prediction - avg_productivity) / avg_productivity) * 100
        
        if diff > 0:
            st.info(f"📈 {abs(diff):.1f}% above national average")
        else:
            st.warning(f"📉 {abs(diff):.1f}% below national average")

def show_trends(df):
    """Trends and insights page"""
    st.header("📈 Trends & Insights")
    
    # Year-over-year growth
    yearly = df.groupby('tahun').agg({
        'produktivitas_ton_per_ha': 'mean',
        'produksi_ton': 'sum',
        'luas_panen_ha': 'sum'
    }).reset_index()
    
    yearly['growth_rate'] = yearly['produktivitas_ton_per_ha'].pct_change() * 100
    
    st.subheader("Year-over-Year Growth Rate")
    fig = px.bar(yearly, x='tahun', y='growth_rate',
                title="Productivity Growth Rate (%)",
                color='growth_rate',
                color_continuous_scale=['red', 'yellow', 'green'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional comparison
    st.subheader("Regional Performance Comparison")
    regional = df.groupby('provinsi')['produktivitas_ton_per_ha'].mean().sort_values(ascending=False).head(15)
    fig = px.bar(regional, orientation='h', 
                title="Top 15 Regions by Average Productivity")
    st.plotly_chart(fig, use_container_width=True)

def generate_sample_data():
    """Generate sample data for demo"""
    import numpy as np
    
    years = [2020, 2021, 2022]
    regions = ['Malang', 'Bandung', 'Bogor', 'Semarang']
    
    data = []
    for year in years:
        for region in regions:
            data.append({
                'tahun': year,
                'region_key': region,
                'provinsi': region,
                'produktivitas_ton_per_ha': np.random.uniform(4.5, 5.5),
                'produksi_ton': np.random.uniform(200000, 300000),
                'luas_panen_ha': np.random.uniform(40000, 60000),
                'suhu_rata_c': np.random.uniform(24, 28),
                'curah_hujan_mm': np.random.uniform(1800, 2200),
                'kelembaban_persen': np.random.uniform(70, 80),
                'hari_hujan': np.random.randint(100, 140)
            })
    
    return pd.DataFrame(data)

# Run app
if __name__ == "__main__":
    main()