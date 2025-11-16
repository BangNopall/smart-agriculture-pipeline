import pandas as pd
import numpy as np
from pathlib import Path

class DataTransformer:
    """Transform dan integrate multiple data sources"""
    
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_bps_data(self, df_bps):
        """Pembersihan data BPS"""
        print("🔧 Cleaning BPS data...")
        
        df = df_bps.copy()
        
        # 1. Handle missing values
        print(f"  Missing values before: {df.isnull().sum().sum()}")
        
        # Imputasi dengan median untuk numeric columns
        numeric_cols = ['luas_panen_ha', 'produksi_ton', 'produktivitas_ton_per_ha']
        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # 2. Remove outliers (IQR method)
        Q1 = df['produktivitas_ton_per_ha'].quantile(0.25)
        Q3 = df['produktivitas_ton_per_ha'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[
            (df['produktivitas_ton_per_ha'] >= lower_bound) & 
            (df['produktivitas_ton_per_ha'] <= upper_bound)
        ]
        
        # 3. Standardize region names
        df['provinsi'] = df['provinsi'].str.strip().str.upper()
        
        print(f"  ✓ Cleaned: {df.shape[0]} rows remaining")
        return df
    
    def clean_weather_data(self, df_weather):
        """Pembersihan data cuaca"""
        print("🔧 Cleaning weather data...")
        
        df = df_weather.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        
        # Standardize region names
        df['provinsi'] = df['provinsi'].str.strip().str.upper()
        
        print(f"  ✓ Cleaned: {df.shape[0]} rows")
        return df
    
    def create_weather_features(self, df_weather):
        """Feature engineering untuk data cuaca"""
        print("🔧 Creating weather features...")
        
        df = df_weather.copy()
        
        # 1. Temperature features
        df['suhu_kategori'] = pd.cut(
            df['suhu_rata_c'],
            bins=[0, 24, 28, 100],
            labels=['dingin', 'optimal', 'panas']
        )
        
        # 2. Rainfall features
        df['hujan_kategori'] = pd.cut(
            df['curah_hujan_mm'],
            bins=[0, 1500, 2500, 10000],
            labels=['rendah', 'normal', 'tinggi']
        )
        
        # 3. Drought index (simplified)
        df['drought_index'] = df['curah_hujan_mm'] / (df['suhu_rata_c'] * 100)
        
        # 4. Humidity stress
        df['humidity_stress'] = np.where(
            (df['kelembaban_persen'] < 60) | (df['kelembaban_persen'] > 90),
            1, 0
        )
        
        print(f"  ✓ Created {len(df.columns) - len(df_weather.columns)} new features")
        return df
    
    def integrate_datasets(self, df_bps, df_weather):
        """Integrate BPS dan BMKG data"""
        print("🔗 Integrating datasets...")
        
        # Join berdasarkan provinsi dan tahun
        df_integrated = pd.merge(
            df_bps,
            df_weather,
            on=['provinsi', 'tahun'],
            how='inner'
        )
        
        print(f"  ✓ Integrated: {df_integrated.shape[0]} rows, {df_integrated.shape[1]} columns")
        
        # Check join quality
        match_rate = len(df_integrated) / len(df_bps) * 100
        print(f"  Match rate: {match_rate:.1f}%")
        
        return df_integrated
    
    def create_final_features(self, df):
        """Create final feature set untuk modeling"""
        print("🔧 Creating final features...")
        
        df_final = df.copy()
        
        # 1. Lag features (previous year's productivity)
        df_final = df_final.sort_values(['provinsi', 'tahun'])
        df_final['produktivitas_prev_year'] = df_final.groupby('provinsi')['produktivitas_ton_per_ha'].shift(1)
        
        # 2. Interaction features
        df_final['curah_hujan_x_suhu'] = df_final['curah_hujan_mm'] * df_final['suhu_rata_c']
        df_final['luas_x_kelembaban'] = df_final['luas_panen_ha'] * df_final['kelembaban_persen']
        
        # 3. Normalized features (0-1 scale)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        cols_to_normalize = ['suhu_rata_c', 'curah_hujan_mm', 'kelembaban_persen']
        df_final[[c + '_norm' for c in cols_to_normalize]] = scaler.fit_transform(df_final[cols_to_normalize])
        
        print(f"  ✓ Final dataset: {df_final.shape}")
        return df_final
    
    def run_full_pipeline(self, df_bps, df_weather):
        """Jalankan full transformation pipeline"""
        print("\n" + "="*60)
        print("STARTING TRANSFORMATION PIPELINE")
        print("="*60)
        
        # Clean data
        df_bps_clean = self.clean_bps_data(df_bps)
        df_weather_clean = self.clean_weather_data(df_weather)
        
        # Feature engineering
        df_weather_feat = self.create_weather_features(df_weather_clean)
        
        # Integration
        df_integrated = self.integrate_datasets(df_bps_clean, df_weather_feat)
        
        # Final features
        df_final = self.create_final_features(df_integrated)
        
        # Save
        output_file = self.processed_dir / "final_dataset.csv"
        df_final.to_csv(output_file, index=False)
        
        print(f"\n✓ Pipeline complete! Saved to {output_file}")
        print(f"Final shape: {df_final.shape}")
        
        return df_final

# Usage
if __name__ == "__main__":
    # Load raw data
    df_bps = pd.read_csv("data/raw/bps_2024_clean.csv")
    df_weather = pd.read_csv("data/raw/bmkg_weather_data.csv")
    
    # Transform
    transformer = DataTransformer()
    df_final = transformer.run_full_pipeline(df_bps, df_weather)
    
    # Show sample
    print("\nSample of final dataset:")
    print(df_final.head())
    print("\nColumns:", df_final.columns.tolist())