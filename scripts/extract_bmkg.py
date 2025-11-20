import pandas as pd
import re
from pathlib import Path

class BMKGExtractor:
    """
    Extract & normalisasi data iklim rata-rata per provinsi (tahun 2024)
    dengan output 34 provinsi (tanpa perluasan provinsi Papua).
    """

    # 34 provinsi sesuai format lama BPS (pra-pemekaran)
    PROVINCES_34 = [
        'ACEH', 'SUMATERA UTARA', 'SUMATERA BARAT', 'RIAU', 'JAMBI',
        'SUMATERA SELATAN', 'BENGKULU', 'LAMPUNG', 'KEP. BANGKA BELITUNG',
        'KEP. RIAU', 'DKI JAKARTA', 'JAWA BARAT', 'JAWA TENGAH',
        'DI YOGYAKARTA', 'JAWA TIMUR', 'BANTEN', 'BALI',
        'NUSA TENGGARA BARAT', 'NUSA TENGGARA TIMUR',
        'KALIMANTAN BARAT', 'KALIMANTAN TENGAH', 'KALIMANTAN SELATAN',
        'KALIMANTAN TIMUR',
        'SULAWESI UTARA', 'SULAWESI TENGAH', 'SULAWESI SELATAN',
        'SULAWESI TENGGARA', 'GORONTALO', 'SULAWESI BARAT',
        'MALUKU', 'MALUKU UTARA',
        'PAPUA', 'PAPUA BARAT'
    ]

    def __init__(
        self,
        input_path: str = "data/raw/rata_rata_provinsi_2024.csv",
        output_path: str = "data/raw/bmkg_weather_data.csv",
        year: int = 2024,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.year = year

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(name: str) -> str:
        """Normalisasi string provinsi untuk mapping."""
        return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

    def _load_raw(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Tidak menemukan file: {self.input_path}"
            )
        df = pd.read_csv(self.input_path)

        required_cols = {"Provinsi", "TAVG", "RR", "RH_AVG"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Kolom berikut harus ada: {missing}")

        return df

    def _normalize_province_names(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Mapping nama provinsi dari file ke format 34 provinsi BPS."""
        bps_map = {self._key(p): p for p in self.PROVINCES_34}

        prov_mapped = []
        for name in df_raw["Provinsi"]:
            key = self._key(name)
            target = bps_map.get(key)

            if not target:
                # fallback: guess uppercase words
                guess = re.sub(r'(?<!^)([A-Z])', r' \1', str(name)).upper()
                target = guess
                print(f"⚠ Provinsi '{name}' tidak ditemukan, pakai '{target}' (fallback)")

            prov_mapped.append(target)

        df = df_raw.copy()
        df["provinsi"] = prov_mapped
        return df

    def extract(self) -> pd.DataFrame:
        print("📥 Extracting & normalizing BMKG climate data (34 provinsi)...")

        df_raw = self._load_raw()
        df_norm = self._normalize_province_names(df_raw)

        # Filter hanya provinsi yang ada di 34-list
        df_norm = df_norm[df_norm["provinsi"].isin(self.PROVINCES_34)]

        df_weather = pd.DataFrame({
            "tahun": self.year,
            "provinsi": df_norm["provinsi"].str.upper(),
            "suhu_rata_c": df_norm["TAVG"],
            "curah_hujan_mm": df_norm["RR"],
            "kelembaban_persen": df_norm["RH_AVG"],
        })

        # Approx hari hujan
        df_weather["hari_hujan"] = (
            (df_weather["curah_hujan_mm"] / 10)
            .clip(lower=50, upper=200)
            .round()
            .astype(int)
        )

        print(f"  ✓ Total provinsi hasil: {df_weather['provinsi'].nunique()}")
        print(f"  ✓ Total baris: {len(df_weather)}")

        return df_weather

    def save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_path, index=False)
        print(f"💾 Disimpan ke: {self.output_path}")


if __name__ == "__main__":
    extractor = BMKGExtractor()
    df_weather = extractor.extract()
    extractor.save(df_weather)

    print("\n📊 Sample:")
    print(df_weather.head())

    print("\n✓ BMKG extraction selesai (34 provinsi).")