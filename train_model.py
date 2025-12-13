"""
Insurance Cost Prediction Model Training Script
Trains a Random Forest model on synthetic insurance data and saves it as a serialized model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

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
    'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL', 'PA'], n_samples),
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
    ((df['location'] == 'NY').astype(int) * 2000) +  # NY is more expensive
    ((df['location'] == 'CA').astype(int) * 1500) +  # CA is moderately expensive
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

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nModel R² Score (Train): {train_score:.4f}")
print(f"Model R² Score (Test): {test_score:.4f}")

# Save model and encoders
model_dir = 'src/main/resources/models'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'insurance_model.pkl')
le_gender_path = os.path.join(model_dir, 'label_encoder_gender.pkl')
le_location_path = os.path.join(model_dir, 'label_encoder_location.pkl')

joblib.dump(model, model_path)
joblib.dump(le_gender, le_gender_path)
joblib.dump(le_location, le_location_path)

print(f"\nModel saved to: {model_path}")
print(f"Gender encoder saved to: {le_gender_path}")
print(f"Location encoder saved to: {le_location_path}")

# Test prediction with sample data
sample_input = np.array([[45, 1, 28.5, 2, 0, 2]])  # age, gender_encoded, bmi, kids, smoker_int, location_encoded
sample_prediction = model.predict(sample_input)[0]
print(f"\nSample prediction for age=45, gender=female, bmi=28.5, kids=2, smoker=False, location=TX: ${sample_prediction:.2f}")
