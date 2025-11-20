import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DataTransformer:
    """Transform dan integrasi data BPS + BMKG (single year: 2024)"""
    
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_bps_data(self, df_bps: pd.DataFrame) -> pd.DataFrame:
        """Pembersihan data BPS"""
        print("🔧 Cleaning BPS data...")
        
        df = df_bps.copy()
        
        # Pastikan nama provinsi konsisten
        df["provinsi"] = df["provinsi"].astype(str).str.strip().str.upper()
        
        # Pastikan kolom numerik benar
        num_cols = ["luas_panen_ha", "produksi_ton", "produktivitas_ton_per_ha"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Tahun harus ada, tapi semua 2024 (sesuai case)
        if "tahun" not in df.columns:
            df["tahun"] = 2024
        
        print(f"  ✓ Cleaned BPS: {df.shape[0]} rows")
        return df
    
    def clean_weather_data(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Pembersihan data cuaca BMKG"""
        print("🔧 Cleaning weather data...")
        
        df = df_weather.copy()
        
        # Standardize provinsi
        df["provinsi"] = df["provinsi"].astype(str).str.strip().str.upper()
        
        # Imputasi numeric sederhana
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                df[col].fillna(df[col].mean(), inplace=True)
        
        # Pastikan kolom tahun ada
        if "tahun" not in df.columns:
            df["tahun"] = 2024
        
        print(f"  ✓ Cleaned weather: {df.shape[0]} rows")
        return df
    
    def create_weather_features(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering untuk data cuaca"""
        print("🔧 Creating weather features...")
        
        df = df_weather.copy()
        
        # 1. Kategori suhu
        df["suhu_kategori"] = pd.cut(
            df["suhu_rata_c"],
            bins=[0, 24, 28, 100],
            labels=["dingin", "optimal", "panas"]
        )
        
        # 2. Kategori curah hujan
        df["hujan_kategori"] = pd.cut(
            df["curah_hujan_mm"],
            bins=[0, 1500, 2500, 10000],
            labels=["rendah", "normal", "tinggi"]
        )
        
        # 3. Drought index (simplified)
        df["drought_index"] = df["curah_hujan_mm"] / (df["suhu_rata_c"] * 100)
        
        # ⚠️ humidity_stress DIHAPUS sesuai permintaan
        # (tidak dibuat lagi di sini)
        
        print(f"  ✓ Weather features added. Total cols: {df.shape[1]}")
        return df
    
    def integrate_datasets(
        self,
        df_bps: pd.DataFrame,
        df_weather: pd.DataFrame
    ) -> pd.DataFrame:
        """Join BPS & BMKG by (provinsi, tahun)"""
        print("🔗 Integrating datasets...")
        
        df_integrated = pd.merge(
            df_bps,
            df_weather,
            on=["provinsi", "tahun"],
            how="inner"
        )
        
        print(f"  ✓ Integrated rows: {df_integrated.shape[0]}")
        print(f"    Unique provinsi: {df_integrated['provinsi'].nunique()}")
        return df_integrated
    
    def create_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fitur akhir untuk modeling.
        TANPA produktivitas_prev_year karena cuma ada tahun 2024.
        """
        print("🔧 Creating final features...")
        
        df_final = df.copy()
        
        # 1. Interaction features (cuaca x panen)
        df_final["curah_hujan_x_suhu"] = (
            df_final["curah_hujan_mm"] * df_final["suhu_rata_c"]
        )
        df_final["luas_x_kelembaban"] = (
            df_final["luas_panen_ha"] * df_final["kelembaban_persen"]
        )
        
        # 2. Normalisasi beberapa fitur cuaca (0–1)
        scaler = MinMaxScaler()
        cols_to_norm = ["suhu_rata_c", "curah_hujan_mm", "kelembaban_persen"]
        
        df_final[[c + "_norm" for c in cols_to_norm]] = scaler.fit_transform(
            df_final[cols_to_norm]
        )
        
        # ⚠️ TIDAK ada lagi:
        # df_final["produktivitas_prev_year"] = ...
        
        print(f"  ✓ Final dataset shape: {df_final.shape}")
        return df_final
    
    def run_full_pipeline(self, df_bps: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Run full pipeline dan simpan final_dataset.csv"""
        print("\n" + "=" * 60)
        print("STARTING TRANSFORMATION PIPELINE (2024 only)")
        print("=" * 60)
        
        df_bps_clean = self.clean_bps_data(df_bps)
        df_weather_clean = self.clean_weather_data(df_weather)
        df_weather_feat = self.create_weather_features(df_weather_clean)
        df_integrated = self.integrate_datasets(df_bps_clean, df_weather_feat)
        df_final = self.create_final_features(df_integrated)
        
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