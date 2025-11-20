import pandas as pd
from pathlib import Path
import sys

def extract_bps_data(file_path: str = "data/raw/bps_produksi_padi.csv") -> pd.DataFrame:
    """
    Extract data produksi padi dari BPS (format 2024 dengan header 4 baris)
    Data hanya untuk tahun 2024, 38 provinsi.
    """
    print("📥 Extracting BPS data...")
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Tidak menemukan file BPS: {file_path}. "
            f"Pastikan bps_produksi_padi.csv sudah diletakkan di folder yang benar."
        )

    # BPS 2024: 4 baris header, tanpa header nama kolom yang formal
    df = pd.read_csv(file_path, skiprows=4, header=None)

    if df.shape[1] != 4:
        raise ValueError(
            f"⚠ Data tidak sesuai, jumlah kolom: {df.shape[1]} (seharusnya 4). "
            f"Contoh baris: 'ACEH,301196.35,55.11,1659966.28'"
        )

    df.columns = ["provinsi", "luas_panen_ha", "produktivitas_ku_ha", "produksi_ton"]

    # Bersihkan dan filter data
    df["provinsi"] = df["provinsi"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["provinsi"])
    df = df[df["provinsi"] != "INDONESIA"]

    for col in ["luas_panen_ha", "produktivitas_ku_ha", "produksi_ton"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["luas_panen_ha", "produktivitas_ku_ha", "produksi_ton"])

    # Konversi ku/ha → ton/ha
    df["produktivitas_ton_per_ha"] = df["produktivitas_ku_ha"] / 10.0

    # Tambah kolom tahun (sesuai problem case)
    df["tahun"] = 2024

    # Urut kolom
    df = df[["tahun", "provinsi", "luas_panen_ha", "produksi_ton", "produktivitas_ton_per_ha"]]

    print("\n📊 Contoh data BPS:")
    print(df.head().to_string(index=False))

    print(f"\n✓ BPS data extracted: {len(df)} provinsi")
    print(f"  Total luas panen: {df['luas_panen_ha'].sum():,.2f} ha")
    print(f"  Total produksi   : {df['produksi_ton'].sum():,.2f} ton")
    print(f"  Rata2 produktivitas: {df['produktivitas_ton_per_ha'].mean():.2f} ton/ha")

    return df


def validate_bps_data(df: pd.DataFrame) -> bool:
    """Validasi simple untuk data BPS"""
    print("\n🔍 Validating BPS data...")

    missing = df.isnull().sum()
    if missing.any():
        print("  ⚠ Missing values:")
        print(missing[missing > 0])
    else:
        print("  ✓ Tidak ada missing values")

    print(
        f"\n  Rentang produktivitas: "
        f"{df['produktivitas_ton_per_ha'].min():.2f} - "
        f"{df['produktivitas_ton_per_ha'].max():.2f} ton/ha"
    )

    print("\n  🏆 Top 5 provinsi (produktif):")
    top5 = df.nlargest(5, "produktivitas_ton_per_ha")[["provinsi", "produktivitas_ton_per_ha"]]
    print(top5.to_string(index=False))

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("BPS DATA EXTRACTION - PADI 2024")
    print("=" * 60)

    path_arg = sys.argv[1] if len(sys.argv) > 1 else "data/raw/bps_produksi_padi.csv"

    df_bps = extract_bps_data(path_arg)
    validate_bps_data(df_bps)

    output_path = Path("data/raw/bps_2024_clean.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_bps.to_csv(output_path, index=False)

    print(f"\n💾 Saved clean BPS data to: {output_path}")
    print("\n✓ BPS extraction complete")
