# Insurance Cost Prediction Models Summary

## Overview
This project now trains **three different machine learning models** using the same insurance dataset:
1. **Random Forest Regressor**
2. **Decision Tree Regressor**
3. **Linear Regression**

All models are trained with the same data and saved separately for comparison and use.

---

## Model Details

### 1. Random Forest Regressor
**File:** `insurance_model.pkl` (default model)

**Algorithm:** Ensemble method that uses multiple decision trees and averages their predictions.

**Performance (Latest Training):**
- **Test R²:** 0.9034 (90.34% variance explained)
- **Test RMSE:** $1,003.44
- **Training Parameters:** 100 estimators, max depth of 10

**Characteristics:**
- Best overall performance
- Robust to overfitting
- Good generalization
- Handles non-linear relationships well

---

### 2. Decision Tree Regressor
**File:** `insurance_model_decision_tree.pkl`

**Algorithm:** Single tree that splits data based on feature values to make predictions.

**Performance (Latest Training):**
- **Test R²:** 0.7858 (78.58% variance explained)
- **Test RMSE:** $1,494.37
- **Training Parameters:** max depth of 10

**Characteristics:**
- Simple and interpretable
- Prone to overfitting (notice high training R² but lower test R²)
- Fast predictions
- Can capture complex patterns but less stable than ensemble methods

---

### 3. Linear Regression
**File:** `insurance_model_linear_regression.pkl`

**Algorithm:** Linear model that finds the best-fit line through the data.

**Performance (Latest Training):**
- **Test R²:** 0.9087 (90.87% variance explained)
- **Test RMSE:** $975.71
- **Training Parameters:** None (standard linear regression)

**Characteristics:**
- Surprisingly best test performance in this case
- Simple and interpretable
- Fast training and prediction
- Assumes linear relationships between features and target
- Lower training R² indicates it's not overfitting

---

## Model Comparison

| Model | Test R² | Test RMSE | Training Time | Complexity |
|-------|---------|-----------|---------------|------------|
| **Random Forest** | 0.9034 | $1,003.44 | Medium | High |
| **Decision Tree** | 0.7858 | $1,494.37 | Fast | Medium |
| **Linear Regression** | 0.9087 | $975.71 | Very Fast | Low |

### Best Model for This Dataset
**Linear Regression** performs best on the test set with:
- Highest Test R²: 0.9087
- Lowest Test RMSE: $975.71
- Fastest prediction time
- No overfitting issues

---

## Sample Predictions

For a 45-year-old female with BMI 28.5, 2 kids, non-smoker, from Texas:

| Model | Predicted Cost |
|-------|----------------|
| Random Forest | $9,081.47 |
| Decision Tree | $8,203.77 |
| Linear Regression | $9,665.75 |

---

## Files Generated

All model files are saved in: `src/main/resources/models/`

```
models/
├── insurance_model.pkl                        # Random Forest (default)
├── insurance_model_decision_tree.pkl          # Decision Tree
├── insurance_model_linear_regression.pkl      # Linear Regression
├── label_encoder_gender.pkl                   # Gender encoder
├── label_encoder_location.pkl                 # Location encoder
└── model_config.json                          # Configuration & metrics
```

---

## Training the Models

Run either training script to train all three models:

```bash
# Basic training with synthetic data
python train_model.py

# Advanced training with options
python train_model_dynamic.py --source synthetic --n_samples 500
```

Both scripts will:
1. Load/generate the dataset
2. Preprocess features
3. Train all three models
4. Evaluate performance
5. Save all models and configurations
6. Display comparison metrics

---

## Configuration File

The `model_config.json` file contains:
- List of available models
- Default model (Random Forest for backward compatibility)
- Feature names
- Encoder information
- Performance metrics for each model
- Training/test sample counts

---

## Libraries Used

| Library | Purpose | Usage in Models |
|---------|---------|-----------------|
| **scikit-learn** | Machine learning algorithms | All three model implementations |
| **numpy** | Numerical operations | Data manipulation, arrays |
| **pandas** | Data handling | Loading and preprocessing CSV data |
| **joblib** | Model serialization | Saving/loading trained models |

### Specific sklearn modules:
- `sklearn.ensemble.RandomForestRegressor` - Random Forest model
- `sklearn.tree.DecisionTreeRegressor` - Decision Tree model
- `sklearn.linear_model.LinearRegression` - Linear Regression model
- `sklearn.preprocessing.LabelEncoder` - Encoding categorical variables
- `sklearn.model_selection.train_test_split` - Splitting data
- `sklearn.metrics` - Model evaluation (R², RMSE, MAE)

---

## Next Steps

To use different models in your application, you can:

1. **Modify the prediction service** to load different model files
2. **Add model selection** as a parameter in the API
3. **Create an ensemble** that combines predictions from all models
4. **Compare models** on new data to select the best performer

---

## Conclusion

All three models are now trained and ready to use. Linear Regression surprisingly outperforms the more complex models on this dataset, showing that simpler models can sometimes be better. The Random Forest remains the default model for backward compatibility with existing code.
