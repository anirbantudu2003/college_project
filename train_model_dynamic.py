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
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
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
    
    def train_models(self, X, y, n_estimators=100, max_depth=10, test_size=0.2):
        """Train multiple models: Random Forest, Decision Tree, and Linear Regression"""
        print(f"\n[*] Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        models = {}
        all_metrics = {}
        
        # 1. Random Forest
        print(f"\n[1] Training Random Forest...")
        print(f"    n_estimators={n_estimators}, max_depth={max_depth}")
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_train_pred = rf_model.predict(X_train)
        rf_test_pred = rf_model.predict(X_test)
        
        rf_metrics = self._calculate_metrics(
            y_train, rf_train_pred, y_test, rf_test_pred,
            len(X_train), len(X_test), n_estimators, max_depth
        )
        models['random_forest'] = rf_model
        all_metrics['random_forest'] = rf_metrics
        
        print(f"    Train R²: {rf_metrics['train_r2']:.4f}, Test R²: {rf_metrics['test_r2']:.4f}")
        print(f"    Test RMSE: ${rf_metrics['test_rmse']:.2f}, Test MAE: ${rf_metrics['test_mae']:.2f}")
        
        # 2. Decision Tree
        print(f"\n[2] Training Decision Tree...")
        print(f"    max_depth={max_depth}")
        dt_model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=42
        )
        dt_model.fit(X_train, y_train)
        dt_train_pred = dt_model.predict(X_train)
        dt_test_pred = dt_model.predict(X_test)
        
        dt_metrics = self._calculate_metrics(
            y_train, dt_train_pred, y_test, dt_test_pred,
            len(X_train), len(X_test), None, max_depth
        )
        models['decision_tree'] = dt_model
        all_metrics['decision_tree'] = dt_metrics
        
        print(f"    Train R²: {dt_metrics['train_r2']:.4f}, Test R²: {dt_metrics['test_r2']:.4f}")
        print(f"    Test RMSE: ${dt_metrics['test_rmse']:.2f}, Test MAE: ${dt_metrics['test_mae']:.2f}")
        
        # 3. Linear Regression
        print(f"\n[3] Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_train_pred = lr_model.predict(X_train)
        lr_test_pred = lr_model.predict(X_test)
        
        lr_metrics = self._calculate_metrics(
            y_train, lr_train_pred, y_test, lr_test_pred,
            len(X_train), len(X_test), None, None
        )
        models['linear_regression'] = lr_model
        all_metrics['linear_regression'] = lr_metrics
        
        print(f"    Train R²: {lr_metrics['train_r2']:.4f}, Test R²: {lr_metrics['test_r2']:.4f}")
        print(f"    Test RMSE: ${lr_metrics['test_rmse']:.2f}, Test MAE: ${lr_metrics['test_mae']:.2f}")
        
        # Print comparison
        print(f"\n[+] Model Comparison (Test Set):")
        for model_name, metrics in all_metrics.items():
            print(f"    {model_name.replace('_', ' ').title()}: R²={metrics['test_r2']:.4f}, RMSE=${metrics['test_rmse']:.2f}")
        
        self.metrics = all_metrics
        return models
    
    def _calculate_metrics(self, y_train, train_pred, y_test, test_pred, 
                          n_train, n_test, n_estimators, max_depth):
        """Calculate metrics for a model"""
        metrics = {
            'train_r2': float(r2_score(y_train, train_pred)),
            'test_r2': float(r2_score(y_test, test_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'test_mae': float(mean_absolute_error(y_test, test_pred)),
            'training_samples': n_train,
            'test_samples': n_test,
            'trained_at': datetime.now().isoformat()
        }
        if n_estimators is not None:
            metrics['n_estimators'] = n_estimators
        if max_depth is not None:
            metrics['max_depth'] = max_depth
        return metrics
    
    def save_models(self, models):
        """Save all models and encoders to pickle files"""
        print(f"\n[*] Saving models and encoders...")
        
        # Save Random Forest (default for backward compatibility)
        rf_path = os.path.join(self.model_dir, 'insurance_model.pkl')
        joblib.dump(models['random_forest'], rf_path)
        print(f"[+] Random Forest saved: {rf_path}")
        
        # Save Decision Tree
        dt_path = os.path.join(self.model_dir, 'insurance_model_decision_tree.pkl')
        joblib.dump(models['decision_tree'], dt_path)
        print(f"[+] Decision Tree saved: {dt_path}")
        
        # Save Linear Regression
        lr_path = os.path.join(self.model_dir, 'insurance_model_linear_regression.pkl')
        joblib.dump(models['linear_regression'], lr_path)
        print(f"[+] Linear Regression saved: {lr_path}")
        
        # Save encoders
        for col, encoder in self.encoders.items():
            encoder_path = os.path.join(self.model_dir, f'label_encoder_{col}.pkl')
            joblib.dump(encoder, encoder_path)
            print(f"[+] Encoder saved: {encoder_path}")
        
        # Save feature names and configuration
        config_path = os.path.join(self.model_dir, 'model_config.json')
        config = {
            'models': ['random_forest', 'decision_tree', 'linear_regression'],
            'default_model': 'random_forest',
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'encoders': list(self.encoders.keys())
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[+] Config saved: {config_path}")
        
        return rf_path
    
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
            
            # Train all models
            models = self.train_models(X, y, n_estimators, max_depth)
            
            # Save all models
            model_path = self.save_models(models)
            
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
