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
        user = db_config['user']
        password = db_config['password']
        host = db_config['host']
        port = db_config['port']
        dbname = db_config['database']
        
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(conn_str)
    
    def create_tables(self):
        """Create tables for raw and processed data"""
        print("🗄️ Creating tables...")
        
        schema_sql = """
        -- Table untuk data BPS raw
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
        
        -- Table untuk data terintegrasi + fitur
        DROP TABLE IF EXISTS integrated_agriculture_data CASCADE;
        CREATE TABLE integrated_agriculture_data (
            id SERIAL PRIMARY KEY,
            tahun INTEGER NOT NULL,
            provinsi VARCHAR(100),
            
            -- BPS
            luas_panen_ha FLOAT,
            produksi_ton FLOAT,
            produktivitas_ton_per_ha FLOAT,
            
            -- Weather
            suhu_rata_c FLOAT,
            curah_hujan_mm FLOAT,
            kelembaban_persen FLOAT,
            hari_hujan INTEGER,
            
            -- Engineered features
            suhu_kategori VARCHAR(20),
            hujan_kategori VARCHAR(20),
            drought_index FLOAT,
            -- humidity_stress INTEGER,          -- ❌ DIHAPUS
            -- produktivitas_prev_year FLOAT,    -- ❌ DIHAPUS
            curah_hujan_x_suhu FLOAT,
            luas_x_kelembaban FLOAT,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
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
        
        -- Indexes
        CREATE INDEX idx_integrated_tahun ON integrated_agriculture_data(tahun);
        CREATE INDEX idx_integrated_provinsi ON integrated_agriculture_data(provinsi);
        CREATE INDEX idx_prediction_tahun ON prediction_results(tahun);
        """
        
        with self.engine.connect() as conn:
            for statement in schema_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
        
        print("  ✓ Tables created successfully")
    
    def load_data(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        """Load DataFrame ke database"""
        print(f"⬆️ Loading data to table '{table_name}'...")
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        print("  ✓ Data loaded")
    

if __name__ == "__main__":
    loader = DatabaseLoader()
    
    # Create tables
    loader.create_tables()
    
    # Load raw & processed
    df_bps = pd.read_csv("data/raw/bps_2024_clean.csv")
    df_weather = pd.read_csv("data/raw/bmkg_weather_data.csv")
    df_final = pd.read_csv("data/processed/final_dataset.csv")
    
    loader.load_data(df_bps, "raw_bps_data")
    loader.load_data(df_weather, "raw_weather_data")
    loader.load_data(df_final, "integrated_agriculture_data")