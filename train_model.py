"""
Insurance Cost Prediction Model Training Script
Trains a Random Forest model on synthetic insurance data and saves it as a serialized model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style('whitegrid')
sns.set_palette('husl')

# Generate synthetic training data
np.random.seed(42)
n_samples = 500

# Create synthetic dataset
data = {
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['male', 'female'], n_samples),
    'bmi': np.random.uniform(15, 50, n_samples),
    'kids': np.random.randint(0, 5, n_samples),
    'smoker': np.random.choice([True, False], n_samples),
    'location': np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n_samples),
}

df = pd.DataFrame(data)

# Generate insurance costs based on features (synthetic ground truth)
# Base cost + age factor + BMI factor + smoker factor + kids factor + location factor
base_cost = 3000
df['cost'] = (
    base_cost +
    (df['age'] * 50) +  # Older age = higher cost
    (df['bmi'] * 100) +  # Higher BMI = higher cost
    (df['kids'] * 500) +  # More kids = higher cost
    (df['smoker'].astype(int) * 5000) +  # Smokers pay much more
    ((df['location'] == 'northeast').astype(int) * 2000) +  # Northeast is more expensive
    ((df['location'] == 'southeast').astype(int) * 1500) +  # Southeast is moderately expensive
    np.random.normal(0, 500, n_samples)  # Add some noise
)

# Ensure positive costs
df['cost'] = df['cost'].clip(lower=500)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nCost statistics:")
print(df['cost'].describe())

# Preprocess: encode categorical variables
le_gender = LabelEncoder()
le_location = LabelEncoder()

df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['location_encoded'] = le_location.fit_transform(df['location'])
df['smoker_int'] = df['smoker'].astype(int)

# Features and target
X = df[['age', 'gender_encoded', 'bmi', 'kids', 'smoker_int', 'location_encoded']]
y = df['cost']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {}
metrics = {}

print("\n" + "="*60)
print("Training Multiple Models")
print("="*60)

# 1. Random Forest
print("\n[1] Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_train_score = r2_score(y_train, rf_train_pred)
rf_test_score = r2_score(y_test, rf_test_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
models['random_forest'] = rf_model
metrics['random_forest'] = {
    'train_r2': rf_train_score,
    'test_r2': rf_test_score,
    'train_rmse': rf_train_rmse,
    'test_rmse': rf_test_rmse,
    'train_mae': rf_train_mae,
    'test_mae': rf_test_mae,
    'test_predictions': rf_test_pred,
    'train_predictions': rf_train_pred
}
print(f"    Train R²: {rf_train_score:.4f}, Test R²: {rf_test_score:.4f}")
print(f"    Train RMSE: ${rf_train_rmse:.2f}, Test RMSE: ${rf_test_rmse:.2f}")
print(f"    Train MAE: ${rf_train_mae:.2f}, Test MAE: ${rf_test_mae:.2f}")

# 2. Decision Tree
print("\n[2] Training Decision Tree...")
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)
dt_train_score = r2_score(y_train, dt_train_pred)
dt_test_score = r2_score(y_test, dt_test_pred)
dt_train_rmse = np.sqrt(mean_squared_error(y_train, dt_train_pred))
dt_test_rmse = np.sqrt(mean_squared_error(y_test, dt_test_pred))
dt_train_mae = mean_absolute_error(y_train, dt_train_pred)
dt_test_mae = mean_absolute_error(y_test, dt_test_pred)
models['decision_tree'] = dt_model
metrics['decision_tree'] = {
    'train_r2': dt_train_score,
    'test_r2': dt_test_score,
    'train_rmse': dt_train_rmse,
    'test_rmse': dt_test_rmse,
    'train_mae': dt_train_mae,
    'test_mae': dt_test_mae,
    'test_predictions': dt_test_pred,
    'train_predictions': dt_train_pred
}
print(f"    Train R²: {dt_train_score:.4f}, Test R²: {dt_test_score:.4f}")
print(f"    Train RMSE: ${dt_train_rmse:.2f}, Test RMSE: ${dt_test_rmse:.2f}")
print(f"    Train MAE: ${dt_train_mae:.2f}, Test MAE: ${dt_test_mae:.2f}")

# 3. Linear Regression
print("\n[3] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)
lr_train_score = r2_score(y_train, lr_train_pred)
lr_test_score = r2_score(y_test, lr_test_pred)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
models['linear_regression'] = lr_model
metrics['linear_regression'] = {
    'train_r2': lr_train_score,
    'test_r2': lr_test_score,
    'train_rmse': lr_train_rmse,
    'test_rmse': lr_test_rmse,
    'train_mae': lr_train_mae,
    'test_mae': lr_test_mae,
    'test_predictions': lr_test_pred,
    'train_predictions': lr_train_pred
}
print(f"    Train R²: {lr_train_score:.4f}, Test R²: {lr_test_score:.4f}")
print(f"    Train RMSE: ${lr_train_rmse:.2f}, Test RMSE: ${lr_test_rmse:.2f}")
print(f"    Train MAE: ${lr_train_mae:.2f}, Test MAE: ${lr_test_mae:.2f}")

print("\n" + "="*60)
print("Model Comparison")
print("="*60)
for model_name, model_metrics in metrics.items():
    print(f"\n{model_name.upper()}:")
    print(f"  Test R²: {model_metrics['test_r2']:.4f}")
    print(f"  Test RMSE: ${model_metrics['test_rmse']:.2f}")
    print(f"  Test MAE: ${model_metrics['test_mae']:.2f}")

# Generate Visualizations
print("\n" + "="*60)
print("Generating Visualizations")
print("="*60)

visualization_dir = 'src/main/resources/static/visualizations'
os.makedirs(visualization_dir, exist_ok=True)

# 1. Metrics Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

model_names = list(metrics.keys())
model_labels = [name.replace('_', ' ').title() for name in model_names]

# R² Score Comparison
r2_scores = [metrics[m]['test_r2'] for m in model_names]
axes[0].bar(model_labels, r2_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('R² Score (Higher is Better)', fontsize=13)
axes[0].set_ylim([0, 1])
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# RMSE Comparison
rmse_scores = [metrics[m]['test_rmse'] for m in model_names]
axes[1].bar(model_labels, rmse_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[1].set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
axes[1].set_title('RMSE (Lower is Better)', fontsize=13)
for i, v in enumerate(rmse_scores):
    axes[1].text(i, v + 20, f'${v:.2f}', ha='center', fontweight='bold')

# MAE Comparison
mae_scores = [metrics[m]['test_mae'] for m in model_names]
axes[2].bar(model_labels, mae_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[2].set_ylabel('MAE ($)', fontsize=12, fontweight='bold')
axes[2].set_title('MAE (Lower is Better)', fontsize=13)
for i, v in enumerate(mae_scores):
    axes[2].text(i, v + 20, f'${v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
metrics_chart_path = os.path.join(visualization_dir, 'model_metrics_comparison.png')
plt.savefig(metrics_chart_path, dpi=300, bbox_inches='tight')
print(f"\n[+] Metrics comparison saved: {metrics_chart_path}")
plt.close()

# 2. Actual vs Predicted - All Models in One Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Actual vs Predicted Values (Test Set)', fontsize=16, fontweight='bold')

colors = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (model_name, color) in enumerate(zip(model_names, colors)):
    y_pred = metrics[model_name]['test_predictions']
    
    # Scatter plot
    axes[idx].scatter(y_test, y_pred, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    axes[idx].set_xlabel('Actual Cost ($)', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Predicted Cost ($)', fontsize=12, fontweight='bold')
    model_label = model_name.replace('_', ' ').title()
    r2 = metrics[model_name]['test_r2']
    axes[idx].set_title(f'{model_label}\nR² = {r2:.4f}', fontsize=13)
    axes[idx].legend(loc='upper left')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
actual_vs_pred_path = os.path.join(visualization_dir, 'actual_vs_predicted.png')
plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches='tight')
print(f"[+] Actual vs Predicted chart saved: {actual_vs_pred_path}")
plt.close()

# 3. Residuals Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Residual Plots (Prediction Errors)', fontsize=16, fontweight='bold')

for idx, (model_name, color) in enumerate(zip(model_names, colors)):
    y_pred = metrics[model_name]['test_predictions']
    residuals = y_test - y_pred
    
    # Residual plot
    axes[idx].scatter(y_pred, residuals, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)
    axes[idx].axhline(y=0, color='black', linestyle='--', lw=2)
    
    # Labels
    axes[idx].set_xlabel('Predicted Cost ($)', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
    model_label = model_name.replace('_', ' ').title()
    axes[idx].set_title(f'{model_label}', fontsize=13)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
residuals_path = os.path.join(visualization_dir, 'residuals_plot.png')
plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
print(f"[+] Residuals plot saved: {residuals_path}")
plt.close()

# 4. Detailed Metrics Table
metrics_df = pd.DataFrame({
    'Model': [m.replace('_', ' ').title() for m in model_names],
    'Train R²': [metrics[m]['train_r2'] for m in model_names],
    'Test R²': [metrics[m]['test_r2'] for m in model_names],
    'Train RMSE': [metrics[m]['train_rmse'] for m in model_names],
    'Test RMSE': [metrics[m]['test_rmse'] for m in model_names],
    'Train MAE': [metrics[m]['train_mae'] for m in model_names],
    'Test MAE': [metrics[m]['test_mae'] for m in model_names]
})

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=metrics_df.round(4).values,
                colLabels=metrics_df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#3498db']*len(metrics_df.columns))

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(len(metrics_df.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(metrics_df) + 1):
    for j in range(len(metrics_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Detailed Model Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
table_path = os.path.join(visualization_dir, 'metrics_table.png')
plt.savefig(table_path, dpi=300, bbox_inches='tight')
print(f"[+] Metrics table saved: {table_path}")
plt.close()

print(f"\n[+] All visualizations saved to: {visualization_dir}")

# Save all models and encoders
model_dir = 'src/main/resources/models'
os.makedirs(model_dir, exist_ok=True)

print("\n" + "="*60)
print("Saving Models")
print("="*60)

# Save Random Forest (default model for backward compatibility)
rf_path = os.path.join(model_dir, 'insurance_model.pkl')
joblib.dump(models['random_forest'], rf_path)
print(f"\n[+] Random Forest saved to: {rf_path}")

# Save Decision Tree
dt_path = os.path.join(model_dir, 'insurance_model_decision_tree.pkl')
joblib.dump(models['decision_tree'], dt_path)
print(f"[+] Decision Tree saved to: {dt_path}")

# Save Linear Regression
lr_path = os.path.join(model_dir, 'insurance_model_linear_regression.pkl')
joblib.dump(models['linear_regression'], lr_path)
print(f"[+] Linear Regression saved to: {lr_path}")

# Save encoders
le_gender_path = os.path.join(model_dir, 'label_encoder_gender.pkl')
le_location_path = os.path.join(model_dir, 'label_encoder_location.pkl')
joblib.dump(le_gender, le_gender_path)
joblib.dump(le_location, le_location_path)
print(f"\n[+] Gender encoder saved to: {le_gender_path}")
print(f"[+] Location encoder saved to: {le_location_path}")

# Save model configuration with all metrics
config_path = os.path.join(model_dir, 'model_config.json')

# Convert numpy types to Python native types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

config = {
    'models': ['random_forest', 'decision_tree', 'linear_regression'],
    'default_model': 'random_forest',
    'feature_names': ['age', 'gender_encoded', 'bmi', 'kids', 'smoker_int', 'location_encoded'],
    'encoders': ['gender', 'location'],
    'metrics': {
        'random_forest': convert_to_serializable(metrics['random_forest']),
        'decision_tree': convert_to_serializable(metrics['decision_tree']),
        'linear_regression': convert_to_serializable(metrics['linear_regression'])
    },
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test))
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"[+] Configuration saved to: {config_path}")

# Test prediction with sample data from all models
print("\n" + "="*60)
print("Sample Predictions")
print("="*60)
sample_input = np.array([[45, 1, 28.5, 2, 0, 2]])  # age, gender_encoded, bmi, kids, smoker_int, location_encoded
print("\nInput: age=45, gender=female, bmi=28.5, kids=2, smoker=False, location=TX")
print("\nPredictions:")
for model_name, model_obj in models.items():
    prediction = model_obj.predict(sample_input)[0]
    print(f"  {model_name.replace('_', ' ').title()}: ${prediction:.2f}")
