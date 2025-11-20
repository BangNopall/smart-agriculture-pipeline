import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

class HarvestPredictor:
    """Model regresi untuk prediksi produktivitas panen (ton/ha)"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.feature_columns = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pilih fitur & buang baris yang missing pada fitur/target"""
        print("🔧 Preparing features...")
        
        feature_cols = [
            # Weather features
            "suhu_rata_c",
            "curah_hujan_mm",
            "kelembaban_persen",
            "hari_hujan",
            
            # Agricultural features
            "luas_panen_ha",
            
            # Engineered features
            "drought_index",
            # "humidity_stress",  # ❌ DIHAPUS
            "curah_hujan_x_suhu",
        ]
        
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom fitur berikut tidak ada di dataset final: {missing}")
        
        df_clean = df.dropna(subset=feature_cols + ["produktivitas_ton_per_ha"])
        
        print(f"  Original: {len(df)} rows")
        print(f"  After cleaning: {len(df_clean)} rows")
        
        self.feature_columns = feature_cols
        return df_clean
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Split train / test"""
        print(f"\n📊 Splitting data (test_size={test_size})...")
        
        X = df[self.feature_columns]
        y = df["produktivitas_ton_per_ha"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples : {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """Train model"""
        print("\n🤖 Training model...")
        
        self.model.fit(X_train, y_train)
        
        # Print coefficients
        print("\n📈 Model Coefficients:")
        for feature, coef in zip(self.feature_columns, self.model.coef_):
            print(f"  {feature:30s}: {coef:+.4f}")
        print(f"  {'Intercept':30s}: {self.model.intercept_:+.4f}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate performance"""
        print("\n📊 Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n🎯 Model Performance:")
        print(f"  MSE  : {mse:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  R²   : {r2:.4f}")
        
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}, y_pred
    
    def plot_results(self, y_test, y_pred):
        """Buat plot evaluasi sederhana"""
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual Productivity (ton/ha)")
        axes[0, 0].set_ylabel("Predicted Productivity (ton/ha)")
        axes[0, 0].set_title("Actual vs Predicted")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0, 1].set_xlabel("Predicted Productivity (ton/ha)")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residual Plot")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual distribution
        axes[1, 0].hist(residuals, bins=15, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Residuals")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance (absolute coef)
        feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "coefficient": np.abs(self.model.coef_),
        }).sort_values("coefficient", ascending=True)
        
        axes[1, 1].barh(
            feature_importance["feature"],
            feature_importance["coefficient"]
        )
        axes[1, 1].set_xlabel("Absolute Coefficient Value")
        axes[1, 1].set_title("Feature Importance")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.model_dir / "model_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved plot to {plot_path}")
        return fig
    
    def save_model(self, version: str = "v1"):
        """Simpan model ke disk"""
        model_path = self.model_dir / f"harvest_predictor_{version}.pkl"
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "version": version,
            },
            model_path
        )
        print(f"\n💾 Model saved to {model_path}")
    
    def load_model(self, version: str = "v1"):
        """Load model dari disk"""
        model_path = self.model_dir / f"harvest_predictor_{version}.pkl"
        data = joblib.load(model_path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        print(f"✓ Model loaded from {model_path}")
        return self.model
    
    def predict_new_data(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """Prediksi untuk data baru"""
        X_new = df_new[self.feature_columns]
        preds = self.model.predict(X_new)
        out = df_new.copy()
        out["predicted_productivity"] = preds
        return out


if __name__ == "__main__":
    print("=" * 60)
    print("HARVEST PREDICTION MODEL TRAINING (2024 only)")
    print("=" * 60)
    
    df = pd.read_csv("data/processed/final_dataset.csv")
    
    predictor = HarvestPredictor()
    df_clean = predictor.prepare_features(df)
    
    X_train, X_test, y_train, y_test = predictor.split_data(df_clean)
    predictor.train(X_train, y_train)
    
    metrics, y_pred = predictor.evaluate(X_test, y_test)
    predictor.plot_results(y_test, y_pred)
    
    predictor.save_model(version="v1")
    
    print("\n✓ MODEL TRAINING COMPLETE")