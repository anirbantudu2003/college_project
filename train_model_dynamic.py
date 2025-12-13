"""
Enhanced Insurance Cost Prediction Model Training Script
Supports both synthetic data and Kaggle datasets with dynamic retraining.

Usage:
  python train_model_dynamic.py --source synthetic [--n_samples 500]
  python train_model_dynamic.py --source kaggle --dataset "mirichoi/insurance"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class InsuranceModelTrainer:
    def __init__(self, model_dir='src/main/resources/models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.encoders = {}
        self.feature_names = None
        self.metrics = {}
        
    def load_synthetic_data(self, n_samples=500):
        """Generate synthetic insurance dataset"""
        print(f"[*] Generating synthetic dataset with {n_samples} samples...")
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['male', 'female'], n_samples),
            'bmi': np.random.uniform(15, 50, n_samples),
            'children': np.random.randint(0, 5, n_samples),
            'smoker': np.random.choice(['yes', 'no'], n_samples),
            'region': np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate realistic insurance charges
        base_cost = 3000
        df['charges'] = (
            base_cost +
            (df['age'] * 50) +
            (df['bmi'] * 100) +
            (df['children'] * 500) +
            ((df['smoker'] == 'yes').astype(int) * 5000) +
            ((df['region'] == 'northeast').astype(int) * 1500) +
            np.random.normal(0, 500, n_samples)
        )
        df['charges'] = df['charges'].clip(lower=500)
        
        return df
    
    def load_kaggle_data(self, dataset_id='mirichoi/insurance'):
        """Download and load real Kaggle insurance dataset"""
        print(f"[*] Loading Kaggle dataset: {dataset_id}")
        try:
            import kaggle
            
            # Download dataset
            dataset_name = dataset_id.split('/')[-1]
            download_path = 'kaggle_data'
            os.makedirs(download_path, exist_ok=True)
            
            print(f"[*] Downloading {dataset_id} from Kaggle...")
            kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=True)
            
            # Find and load CSV file
            csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {download_path}")
            
            csv_path = os.path.join(download_path, csv_files[0])
            print(f"[*] Loading CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Ensure required columns exist
            required_cols = ['age', 'gender', 'bmi', 'children', 'smoker', 'region', 'charges']
            if 'charges' not in df.columns:
                # Try alternative column names
                if 'cost' in df.columns:
                    df['charges'] = df['cost']
                elif 'price' in df.columns:
                    df['charges'] = df['price']
                else:
                    raise ValueError("Target column (charges/cost/price) not found")
            
            print(f"[+] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"\nDataset Info:")
            print(df.info())
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            return df
            
        except ImportError:
            print("[!] kaggle package not installed. Install with: pip install kaggle")
            print("[!] Also setup Kaggle API credentials: https://github.com/Kaggle/kaggle-api")
            raise
        except Exception as e:
            print(f"[!] Error downloading Kaggle dataset: {e}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess and encode categorical variables"""
        print("\n[*] Preprocessing data...")
        
        df = df.copy()
        
        # Handle categorical columns
        categorical_cols = ['gender', 'smoker', 'region']
        if 'location' in df.columns:
            categorical_cols.append('location')
        
        # Only encode columns that exist
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        # Select feature columns
        numeric_cols = ['age', 'bmi', 'children']
        encoded_cols = [f'{col}_encoded' for col in categorical_cols]
        feature_cols = numeric_cols + encoded_cols
        
        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df['charges']
        
        print(f"[+] Features: {feature_cols}")
        print(f"[+] Target: charges")
        print(f"[+] Data shape: X={X.shape}, y={y.shape}")
        
        return X, y, df
    
    def train_model(self, X, y, n_estimators=100, max_depth=10, test_size=0.2):
        """Train Random Forest model"""
        print(f"\n[*] Training Random Forest model...")
        print(f"    n_estimators={n_estimators}, max_depth={max_depth}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        self.metrics = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'trained_at': datetime.now().isoformat()
        }
        
        print(f"\n[+] Model Performance:")
        print(f"    Train R²:  {train_r2:.4f}")
        print(f"    Test R²:   {test_r2:.4f}")
        print(f"    Train RMSE: ${train_rmse:.2f}")
        print(f"    Test RMSE:  ${test_rmse:.2f}")
        print(f"    Train MAE:  ${train_mae:.2f}")
        print(f"    Test MAE:   ${test_mae:.2f}")
        
        return model, train_r2, test_r2
    
    def save_model(self, model):
        """Save model and encoders to pickle files"""
        print(f"\n[*] Saving model and encoders...")
        
        # Save model
        model_path = os.path.join(self.model_dir, 'insurance_model.pkl')
        joblib.dump(model, model_path)
        print(f"[+] Model saved: {model_path}")
        
        # Save encoders
        for col, encoder in self.encoders.items():
            encoder_path = os.path.join(self.model_dir, f'label_encoder_{col}.pkl')
            joblib.dump(encoder, encoder_path)
            print(f"[+] Encoder saved: {encoder_path}")
        
        # Save feature names
        config_path = os.path.join(self.model_dir, 'model_config.json')
        config = {
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'encoders': list(self.encoders.keys())
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[+] Config saved: {config_path}")
        
        return model_path
    
    def train(self, source='synthetic', dataset_id=None, n_samples=500, n_estimators=100, max_depth=10):
        """Complete training pipeline"""
        print(f"\n{'='*60}")
        print(f"Insurance Model Training Pipeline")
        print(f"{'='*60}")
        
        try:
            # Load data
            if source == 'kaggle' and dataset_id:
                df = self.load_kaggle_data(dataset_id)
            else:
                df = self.load_synthetic_data(n_samples)
            
            # Preprocess
            X, y, df_processed = self.preprocess_data(df)
            
            # Train
            model, train_r2, test_r2 = self.train_model(X, y, n_estimators, max_depth)
            
            # Save
            model_path = self.save_model(model)
            
            print(f"\n[+] Training completed successfully!")
            print(f"{'='*60}\n")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'metrics': self.metrics
            }
            
        except Exception as e:
            print(f"\n[!] Error during training: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    parser = argparse.ArgumentParser(description='Train insurance prediction model')
    parser.add_argument('--source', choices=['synthetic', 'kaggle'], default='synthetic',
                        help='Data source: synthetic or kaggle')
    parser.add_argument('--dataset', default='mirichoi/insurance',
                        help='Kaggle dataset ID (e.g., mirichoi/insurance)')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum tree depth')
    
    args = parser.parse_args()
    
    trainer = InsuranceModelTrainer()
    result = trainer.train(
        source=args.source,
        dataset_id=args.dataset,
        n_samples=args.n_samples,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
