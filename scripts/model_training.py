import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class HarvestPredictor:
    """Model regresi untuk prediksi produktivitas panen"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.feature_columns = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def prepare_features(self, df):
        """Prepare features untuk modeling"""
        print("🔧 Preparing features...")
        
        # Select feature columns
        feature_cols = [
            # Weather features
            'suhu_rata_c',
            'curah_hujan_mm',
            'kelembaban_persen',
            'hari_hujan',
            
            # Agricultural features
            'luas_panen_ha',
            
            # Engineered features
            'drought_index',
            'humidity_stress',
            'curah_hujan_x_suhu',
        ]
        
        # Remove rows with missing values in key columns
        df_clean = df.dropna(subset=feature_cols + ['produktivitas_ton_per_ha'])
        
        print(f"  Original: {len(df)} rows")
        print(f"  After cleaning: {len(df_clean)} rows")
        
        self.feature_columns = feature_cols
        return df_clean
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data menjadi train dan test"""
        print(f"\n📊 Splitting data (test_size={test_size})...")
        
        X = df[self.feature_columns]
        y = df['produktivitas_ton_per_ha']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
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
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        print("\n🎯 Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric:20s}: {value:.4f}")
        
        # Interpretation
        print("\n💡 Interpretation:")
        print(f"  - Model explains {r2*100:.1f}% of variance in productivity")
        print(f"  - Average prediction error: ±{mae:.2f} ton/ha")
        
        return metrics, y_pred
    
    def plot_results(self, y_test, y_pred):
        """Visualize prediction results"""
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Productivity (ton/ha)')
        axes[0, 0].set_ylabel('Predicted Productivity (ton/ha)')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Productivity (ton/ha)')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': np.abs(self.model.coef_)
        }).sort_values('coefficient', ascending=True)
        
        axes[1, 1].barh(feature_importance['feature'], feature_importance['coefficient'])
        axes[1, 1].set_xlabel('Absolute Coefficient Value')
        axes[1, 1].set_title('Feature Importance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / "model_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {plot_path}")
        
        return fig
    
    def save_model(self, version='v1'):
        """Save trained model"""
        model_path = self.model_dir / f"harvest_predictor_{version}.pkl"
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'version': version
        }
        
        joblib.dump(model_data, model_path)
        print(f"\n💾 Model saved to {model_path}")
    
    def load_model(self, version='v1'):
        """Load trained model"""
        model_path = self.model_dir / f"harvest_predictor_{version}.pkl"
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        
        print(f"✓ Model loaded from {model_path}")
        return self.model
    
    def predict_new_data(self, df_new):
        """Make predictions on new data"""
        X_new = df_new[self.feature_columns]
        predictions = self.model.predict(X_new)
        
        df_result = df_new.copy()
        df_result['predicted_productivity'] = predictions
        
        return df_result

# Main training pipeline
if __name__ == "__main__":
    print("="*60)
    print("HARVEST PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = pd.read_csv("data/processed/final_dataset.csv")
    
    # Initialize predictor
    predictor = HarvestPredictor()
    
    # Prepare features
    df_clean = predictor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(df_clean)
    
    # Train model
    predictor.train(X_train, y_train)
    
    # Evaluate
    metrics, y_pred = predictor.evaluate(X_test, y_test)
    
    # Plot results
    predictor.plot_results(y_test, y_pred)
    
    # Save model
    predictor.save_model(version='v1')
    
    print("\n" + "="*60)
    print("✓ MODEL TRAINING COMPLETE!")
    print("="*60)