import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import yaml

class DatabaseLoader:
    """Load data ke PostgreSQL database"""
    
    def __init__(self, config_path="config/config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        db_config = config['database']
        
        # Create connection string
        self.connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        self.engine = create_engine(self.connection_string)
    
    def create_tables(self):
        """Create database schema"""
        print("📊 Creating database tables...")
        
        schema_sql = """
        -- Table untuk data mentah BPS
        DROP TABLE IF EXISTS raw_bps_data CASCADE;
        CREATE TABLE raw_bps_data (
            id SERIAL PRIMARY KEY,
            tahun INTEGER NOT NULL,
            provinsi VARCHAR(100),
            luas_panen_ha FLOAT,
            produksi_ton FLOAT,
            produktivitas_ton_per_ha FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Table untuk data cuaca
        DROP TABLE IF EXISTS raw_weather_data CASCADE;
        CREATE TABLE raw_weather_data (
            id SERIAL PRIMARY KEY,
            tahun INTEGER NOT NULL,
            provinsi VARCHAR(100),
            suhu_rata_c FLOAT,
            curah_hujan_mm FLOAT,
            kelembaban_persen FLOAT,
            hari_hujan INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Table untuk data terintegrasi (final)
        DROP TABLE IF EXISTS integrated_agriculture_data CASCADE;
        CREATE TABLE integrated_agriculture_data (
            id SERIAL PRIMARY KEY,
            tahun INTEGER NOT NULL,
            provinsi VARCHAR(100),
            
            -- Agricultural data
            luas_panen_ha FLOAT,
            produksi_ton FLOAT,
            produktivitas_ton_per_ha FLOAT,
            
            -- Weather data
            suhu_rata_c FLOAT,
            curah_hujan_mm FLOAT,
            kelembaban_persen FLOAT,
            hari_hujan INTEGER,
            
            -- Engineered features
            suhu_kategori VARCHAR(20),
            hujan_kategori VARCHAR(20),
            drought_index FLOAT,
            humidity_stress INTEGER,
            produktivitas_prev_year FLOAT,
            curah_hujan_x_suhu FLOAT,
            luas_x_kelembaban FLOAT,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Index untuk query performa
            CONSTRAINT unique_year_provinsi UNIQUE (tahun, provinsi)
        );
        
        -- Table untuk hasil prediksi
        DROP TABLE IF EXISTS prediction_results CASCADE;
        CREATE TABLE prediction_results (
            id SERIAL PRIMARY KEY,
            tahun INTEGER NOT NULL,
            provinsi VARCHAR(100),
            actual_productivity FLOAT,
            predicted_productivity FLOAT,
            error_rate FLOAT,
            model_version VARCHAR(50),
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes
        CREATE INDEX idx_integrated_tahun ON integrated_agriculture_data(tahun);
        CREATE INDEX idx_integrated_provinsi ON integrated_agriculture_data(provinsi);
        CREATE INDEX idx_prediction_tahun ON prediction_results(tahun);
        """
        
        with self.engine.connect() as conn:
            for statement in schema_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
        
        print("  ✓ Tables created successfully")
    
    def load_data(self, df, table_name, if_exists='replace'):
        """Load DataFrame to database"""
        print(f"📥 Loading data to {table_name}...")
        
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            print(f"  ✓ Loaded {len(df)} rows to {table_name}")
            return True
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            return False
    
    def verify_data(self):
        """Verify data yang sudah di-load"""
        print("\n🔍 Verifying loaded data...")
        
        verification_queries = {
            'raw_bps_data': "SELECT COUNT(*) as count FROM raw_bps_data",
            'raw_weather_data': "SELECT COUNT(*) as count FROM raw_weather_data",
            'integrated_agriculture_data': "SELECT COUNT(*) as count FROM integrated_agriculture_data"
        }
        
        with self.engine.connect() as conn:
            for table, query in verification_queries.items():
                result = conn.execute(text(query)).fetchone()
                print(f"  {table}: {result[0]} rows")
    
    def create_views(self):
        """Create useful views untuk analytics"""
        print("\n📊 Creating analytical views...")
        
        views_sql = """
        -- View: Produktivitas per provinsi per tahun
        CREATE OR REPLACE VIEW v_productivity_by_province AS
        SELECT 
            tahun,
            provinsi,
            AVG(produktivitas_ton_per_ha) as avg_productivity,
            SUM(produksi_ton) as total_production,
            SUM(luas_panen_ha) as total_area
        FROM integrated_agriculture_data
        GROUP BY tahun, provinsi
        ORDER BY tahun, provinsi;
        
        -- View: Pengaruh cuaca terhadap produktivitas
        CREATE OR REPLACE VIEW v_weather_impact AS
        SELECT 
            tahun,
            hujan_kategori,
            suhu_kategori,
            AVG(produktivitas_ton_per_ha) as avg_productivity,
            COUNT(*) as sample_count
        FROM integrated_agriculture_data
        GROUP BY tahun, hujan_kategori, suhu_kategori
        ORDER BY tahun, hujan_kategori, suhu_kategori;
        
        -- View: Trend produktivitas
        CREATE OR REPLACE VIEW v_productivity_trend AS
        SELECT 
            tahun,
            provinsi AS region_key,
            produktivitas_ton_per_ha,
            LAG(produktivitas_ton_per_ha) OVER (PARTITION BY provinsi ORDER BY tahun) as prev_year_productivity,
            produktivitas_ton_per_ha - LAG(produktivitas_ton_per_ha) OVER (PARTITION BY provinsi ORDER BY tahun) as productivity_change
        FROM integrated_agriculture_data
        ORDER BY provinsi, tahun;
        """
        
        with self.engine.connect() as conn:
            for statement in views_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
        
        print("  ✓ Views created successfully")

# Usage
if __name__ == "__main__":
    # Initialize loader
    loader = DatabaseLoader()
    
    # Create tables
    loader.create_tables()
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA TO DATABASE")
    print("="*60)
    
    # Load raw BPS data
    df_bps = pd.read_csv("data/raw/bps_2024_clean.csv")
    loader.load_data(df_bps, 'raw_bps_data')
    
    # Load raw weather data
    df_weather = pd.read_csv("data/raw/bmkg_weather_data.csv")
    loader.load_data(df_weather, 'raw_weather_data')
    
    # Load integrated data
    df_integrated = pd.read_csv("data/processed/final_dataset.csv")
    loader.load_data(df_integrated, 'integrated_agriculture_data')
    
    # Create views
    loader.create_views()
    
    # Verify
    loader.verify_data()
    
    print("\n✓ All data loaded successfully!")