import pandas as pd
from pathlib import Path

def extract_bps_data(file_path="data/raw/bps_produksi_padi.csv"):
    """
    Extract data produksi padi dari BPS (format 2024 dengan header 4 baris)
    Data hanya untuk tahun 2024, 38 provinsi
    """
    print("📥 Extracting BPS data...")
    
    # Read CSV dengan skip header rows dan tanpa header resmi
    df = pd.read_csv(file_path, skiprows=4, header=None)
    
    # Pastikan kolom benar (4 kolom: provinsi, luas_panen, produktivitas, produksi)
    if df.shape[1] != 4:
        raise ValueError(
            f"⚠ Data tidak sesuai, jumlah kolom: {df.shape[1]} (harusnya 4). "
            f"Pastikan format CSV benar. Contoh baris: 'ACEH,301196.35,55.11,1659966.28'"
        )
    
    # Tetapkan nama kolom
    df.columns = ['provinsi', 'luas_panen_ha', 'produktivitas_ku_ha', 'produksi_ton']
    
    # Bersihkan provinsi
    df['provinsi'] = df['provinsi'].str.strip().str.upper()
    
    # Remove rows invalid (provinsi kosong)
    df = df.dropna(subset=['provinsi'])
    
    # Konversi ke numeric
    for col in ['luas_panen_ha', 'produktivitas_ku_ha', 'produksi_ton']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Hapus baris yang tidak valid (contoh total INDONESIA atau null)
    df = df[df['provinsi'] != 'INDONESIA']
    df = df.dropna(subset=['luas_panen_ha', 'produktivitas_ku_ha', 'produksi_ton'])
    
    # Konversi ku/ha ke ton/ha
    df['produktivitas_ton_per_ha'] = df['produktivitas_ku_ha'] / 10
    
    # Tambah kolom tahun
    df['tahun'] = 2024
    
    # Urut kolom
    df = df[['tahun', 'provinsi', 'luas_panen_ha', 'produksi_ton', 'produktivitas_ton_per_ha']]
    
    # Debug per provinsi
    print("\n📊 Contoh data provinsi yang berhasil dibaca:")
    print(df.head().to_string(index=False))
    
    print(f"\n✓ BPS data extracted: {len(df)} provinsi")
    print(f"  Luas Panen Total: {df['luas_panen_ha'].sum():,.2f} ha")
    print(f"  Produksi Total: {df['produksi_ton'].sum():,.2f} ton")
    print(f"  Rata-rata Produktivitas: {df['produktivitas_ton_per_ha'].mean():.2f} ton/ha")
    
    return df

def validate_bps_data(df):
    """Validasi data BPS"""
    print("\n🔍 Validating BPS data...")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("  ⚠ Missing values found:")
        print(missing[missing > 0])
    else:
        print("  ✓ No missing values")
    
    # Check data ranges
    print(f"\n  Produktivitas range: {df['produktivitas_ton_per_ha'].min():.2f} - {df['produktivitas_ton_per_ha'].max():.2f} ton/ha")
    
    # Check for outliers (provinces with very low productivity)
    low_productivity = df[df['produktivitas_ton_per_ha'] < 3.0]
    if not low_productivity.empty:
        print(f"\n  ⚠ Provinsi dengan produktivitas rendah (<3.0 ton/ha):")
        print(low_productivity[['provinsi', 'produktivitas_ton_per_ha']])
    
    # Top 5 provinces
    print(f"\n  🏆 Top 5 Provinsi (Produktivitas):")
    top5 = df.nlargest(5, 'produktivitas_ton_per_ha')[['provinsi', 'produktivitas_ton_per_ha']]
    print(top5.to_string(index=False))
    
    return True

def expand_to_historical_data(df_2024, start_year=2020):
    """
    Expand data 2024 ke historical data (2020-2024)
    dengan asumsi pertumbuhan organik
    
    CATATAN: Ini adalah simulasi karena data asli hanya 2024
    Untuk production, gunakan data historis yang sebenarnya
    """
    print(f"\n📊 Generating historical data ({start_year}-2024)...")
    
    all_data = []
    
    for year in range(start_year, 2025):
        df_year = df_2024.copy()
        df_year['tahun'] = year
        
        # Simulasi variasi historis (semakin lama semakin kecil)
        year_factor = 1 - ((2024 - year) * 0.02)  # -2% per tahun ke belakang
        
        # Add small random variation
        import numpy as np
        np.random.seed(year)  # Consistent per year
        variation = np.random.normal(1, 0.03, len(df_year))  # ±3% variation
        
        df_year['luas_panen_ha'] = df_year['luas_panen_ha'] * year_factor * variation
        df_year['produksi_ton'] = df_year['produksi_ton'] * year_factor * variation
        df_year['produktivitas_ton_per_ha'] = df_year['produksi_ton'] / df_year['luas_panen_ha']
        
        all_data.append(df_year)
    
    df_historical = pd.concat(all_data, ignore_index=True)
    
    print(f"  ✓ Generated: {len(df_historical)} records ({start_year}-2024)")
    print(f"  Years: {sorted(df_historical['tahun'].unique())}")
    
    return df_historical

if __name__ == "__main__":
    print("="*60)
    print("BPS DATA EXTRACTION - PADI 2024")
    print("="*60)
    
    # Extract data 2024
    df_bps = extract_bps_data("data/raw/bps_produksi_padi.csv")
    
    # Validate
    validate_bps_data(df_bps)
    
    # Save raw 2024 data
    output_path = Path("data/raw/bps_2024_clean.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_bps.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    # Generate historical data for modeling
    df_historical = expand_to_historical_data(df_bps, start_year=2020)
    
    # Save historical data
    historical_path = Path("data/raw/bps_produksi_padi_historical.csv")
    df_historical.to_csv(historical_path, index=False)
    print(f"💾 Saved historical data to: {historical_path}")
    
    print("\n" + "="*60)
    print("✓ BPS EXTRACTION COMPLETE")
    print("="*60)