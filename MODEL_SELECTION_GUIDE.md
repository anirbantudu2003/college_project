# Model Selection Guide

## Overview
The application now supports **three different machine learning models** for insurance cost prediction. Users can select which model to use from the frontend and compare predictions across different models.

---

## Available Models

### 1. **Random Forest** (Default)
- **Best for:** General purpose predictions with good accuracy
- **Performance:** Test R² = 91.04%, RMSE = $940
- **Characteristics:** Robust, handles non-linear relationships well

### 2. **Decision Tree**
- **Best for:** Fast predictions and interpretability
- **Performance:** Test R² = 81.85%, RMSE = $1,338
- **Characteristics:** Simple, fast, but may overfit

### 3. **Linear Regression**
- **Best for:** Understanding linear relationships and fast predictions
- **Performance:** Test R² = 90.53%, RMSE = $966
- **Characteristics:** Simple, interpretable, good generalization

---

## How to Use Model Selection

### Frontend (Web Interface)

1. **Open the application** in your browser: `http://localhost:8080`

2. **Fill in the insurance details:**
   - Name
   - Age
   - Gender (Male/Female/Other)
   - BMI
   - Number of Kids
   - Smoker (Yes/No)
   - Region (Northeast, Southeast, Southwest, Northwest)
   - Existing Conditions (optional)

3. **Select a Prediction Model** from the dropdown:
   - Random Forest (Default)
   - Decision Tree
   - Linear Regression

4. **Click "Get Estimate"** to see the prediction

5. **The result will show:**
   - Which model was used
   - Predicted annual insurance cost
   - Personalized suggestions to reduce cost

---

## Example Request

### Using the Web Form

```
Name: John Doe
Age: 45
Gender: Female
BMI: 28.5
Number of Kids: 2
Smoker: No
Region: Northeast
Prediction Model: Random Forest  ← Select from dropdown
```

### Using API (cURL)

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "age": 45,
    "gender": "female",
    "bmi": 28.5,
    "kids": 2,
    "smoker": false,
    "location": "northeast",
    "model": "random_forest"
  }'
```

### Using API (Different Models)

**Random Forest:**
```json
{
  "age": 45,
  "gender": "female",
  "bmi": 28.5,
  "kids": 2,
  "smoker": false,
  "location": "northeast",
  "model": "random_forest"
}
```

**Decision Tree:**
```json
{
  "age": 45,
  "gender": "female",
  "bmi": 28.5,
  "kids": 2,
  "smoker": false,
  "location": "northeast",
  "model": "decision_tree"
}
```

**Linear Regression:**
```json
{
  "age": 45,
  "gender": "female",
  "bmi": 28.5,
  "kids": 2,
  "smoker": false,
  "location": "northeast",
  "model": "linear_regression"
}
```

---

## Sample Predictions (45-year-old female, BMI 28.5, 2 kids, non-smoker, Northeast)

| Model | Predicted Cost |
|-------|----------------|
| **Random Forest** | $10,683.57 |
| **Decision Tree** | $10,670.53 |
| **Linear Regression** | $10,669.31 |

All three models provide similar predictions for this case, showing consistency across different algorithms.

---

## Response Format

The API returns a response with the model name and prediction:

```
=== Insurance Cost Estimate ===
Model Used: RANDOM FOREST
Estimated Annual Cost: $10,683.57

=== Personalized Suggestions to Reduce Cost ===
1. Maintain your non-smoker status - this keeps your premiums lower
2. Maintain a healthy BMI to keep insurance costs stable
3. Schedule regular health check-ups to prevent chronic conditions
```

---

## Technical Details

### Backend (Java Service)

The `MLPredictionService` class handles model selection:

```java
Map<String, Object> predictionResult = mlPredictionService.predictInsuranceCost(
    age, gender, bmi, kids, smoker, location, selectedModel
);
```

Returns a Map containing:
- `"model"`: Name of the model used
- `"prediction"`: Predicted insurance cost

### Python Script

The `predict_model.py` script accepts a `--model` parameter:

```bash
python predict_model.py 45 female 28.5 2 no northeast --model random_forest
```

Returns JSON output:
```json
{"model": "random_forest", "prediction": 10683.567777632392}
```

### Model Files

All models are stored in `src/main/resources/models/`:

```
models/
├── insurance_model.pkl                        # Random Forest
├── insurance_model_decision_tree.pkl          # Decision Tree
├── insurance_model_linear_regression.pkl      # Linear Regression
├── label_encoder_gender.pkl                   # Gender encoder
├── label_encoder_location.pkl                 # Location encoder
└── model_config.json                          # Configuration
```

---

## Comparing Models

### When to Use Each Model

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| **General use** | Random Forest | Best balance of accuracy and robustness |
| **Need fast predictions** | Linear Regression | Fastest, good accuracy |
| **Want interpretability** | Decision Tree | Easy to understand decision rules |
| **Maximum accuracy** | Random Forest | Highest test R² score |
| **Minimal overfitting** | Linear Regression | Good generalization |

### Performance Comparison

```
Model               Test R²    Test RMSE    Speed      Complexity
─────────────────────────────────────────────────────────────────
Random Forest       91.04%     $940        Medium      High
Decision Tree       81.85%     $1,338      Fast        Medium
Linear Regression   90.53%     $966        Very Fast   Low
```

---

## Troubleshooting

### Model Selection Not Working

1. **Check model parameter is being sent:**
   ```javascript
   console.log(payload.model); // Should print: random_forest, decision_tree, or linear_regression
   ```

2. **Verify model files exist:**
   ```bash
   ls src/main/resources/models/
   ```

3. **Check backend logs:**
   ```
   [ML Service] Successfully predicted: $10683.57 using random_forest
   ```

### Different Models Give Same Result

- This is normal if the input data is similar
- Try different inputs (e.g., smoker vs non-smoker) to see variations

### Invalid Model Error

- Ensure model name is one of: `random_forest`, `decision_tree`, `linear_regression`
- Model names are case-sensitive (use lowercase with underscores)

---

## Future Enhancements

Potential improvements:
1. **Model comparison view** - Show predictions from all models side-by-side
2. **Model confidence intervals** - Display prediction uncertainty
3. **Feature importance** - Show which factors impact cost most for each model
4. **Ensemble predictions** - Combine all three models for better accuracy
5. **Model performance tracking** - Track which model performs best over time

---

## Summary

✅ **Frontend dropdown** allows users to select prediction model  
✅ **Three trained models** available (Random Forest, Decision Tree, Linear Regression)  
✅ **Response shows which model was used** for transparency  
✅ **All models trained on same data** for fair comparison  
✅ **Easy to extend** with additional models in the future  

The model selection feature gives users flexibility and transparency in how their insurance cost is predicted!
