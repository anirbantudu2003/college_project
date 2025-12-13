# Insurance Cost Prediction Application

A full-stack application that uses a machine learning model trained with scikit-learn to predict medical insurance costs based on user input. The application combines a Spring Boot backend, a trained Random Forest model, and a modern HTML frontend.

## Architecture

```
Frontend (HTML/JS) → Spring Boot REST API → ML Model (Python/scikit-learn)
                          ↓
                  InsuranceRequest DTO
                          ↓
                  PredictController
                          ↓
                  MLPredictionService (calls Python subprocess)
                          ↓
                  Trained Random Forest Model (insurance_model.pkl)
```

## Features

- **Form-based input**: Users provide 8 parameters to estimate insurance costs
- **Machine Learning prediction**: Uses a trained Random Forest model for accurate cost estimation
- **Personalized suggestions**: AI-generated suggestions based on user health profile
- **REST API**: Clean JSON API for frontend-backend communication
- **Model persistence**: Trained model saved as pickle files for quick loading

## Setup Instructions

### 1. Python Environment & Model Training

The project includes a Python training script that builds and saves the ML model.

#### Install Python dependencies:
```powershell
# Using pip (if no venv exists)
pip install numpy pandas scikit-learn joblib

# Or configure the existing venv
."\.venv\Scripts\Activate.ps1"
pip install numpy pandas scikit-learn joblib
```

#### Train the model:
```powershell
# From the project root directory
& ".\.venv\Scripts\python.exe" train_model.py
```

This script will:
- Generate synthetic insurance data (500 samples)
- Train a Random Forest regressor
- Save the trained model to `src/main/resources/models/insurance_model.pkl`
- Save label encoders for categorical features
- Print model performance metrics

**Output files created:**
- `src/main/resources/models/insurance_model.pkl` — trained Random Forest model
- `src/main/resources/models/label_encoder_gender.pkl` — encoder for gender
- `src/main/resources/models/label_encoder_location.pkl` — encoder for location

### 2. Java/Spring Boot Setup

#### Build the application:
```powershell
# From the project root
.\mvnw.cmd clean install
```

#### Run the application:
```powershell
.\mvnw.cmd spring-boot:run
```

The application will start on `http://localhost:8080`

### 3. Frontend

Once the app is running, open:
```
http://localhost:8080/
```

The static HTML form will be served from `src/main/resources/static/index.html`

## Form Fields

The frontend collects the following inputs:

| Field | Type | Example |
|-------|------|---------|
| Name | Text | John Doe |
| Age | Number | 45 |
| Gender | Select | Male / Female / Other |
| Smoker | Checkbox | Checked/Unchecked |
| Location | Text | NY, CA, TX, FL, IL, PA |
| Number of Kids | Number | 2 |
| BMI | Decimal | 28.5 |
| Existing Conditions | Text Area | hypertension, diabetes |

## API Endpoint

### POST /api/predict

**Request:**
```json
{
  "name": "John Doe",
  "age": 45,
  "gender": "male",
  "smoker": false,
  "location": "NY",
  "kids": 2,
  "bmi": 28.5,
  "existingCondition": "hypertension"
}
```

**Response:**
```json
{
  "result": "=== Insurance Cost Estimate ===\nEstimated Annual Cost: $12,345.67\n\n=== Personalized Suggestions to Reduce Cost ===\n1. ...\n2. ...\n3. ..."
}
```

## Project Structure

```
demo/
├── train_model.py                      # Model training script
├── src/main/
│   ├── java/com/example/demo/
│   │   ├── controller/
│   │   │   ├── Home.java              # Gemini API controller (optional)
│   │   │   └── PredictController.java # ML prediction endpoint
│   │   ├── service/
│   │   │   ├── GeminiService.java     # Gemini API client (optional)
│   │   │   └── MLPredictionService.java # ML model wrapper
│   │   └── model/
│   │       ├── InsuranceRequest.java  # Input DTO
│   │       └── InsuranceResponse.java # Output DTO
│   └── resources/
│       ├── application.properties     # Spring config
│       ├── static/index.html          # Frontend form
│       ├── models/
│       │   ├── insurance_model.pkl
│       │   ├── label_encoder_gender.pkl
│       │   └── label_encoder_location.pkl
│       └── predict_model.py           # Python prediction helper
├── pom.xml                            # Maven configuration
└── mvnw.cmd                           # Maven wrapper (Windows)
```

## Model Details

### Training Data
- **Samples**: 500 synthetic records
- **Target**: Annual insurance cost (USD)
- **Range**: $500 - $20,000

### Feature Engineering
- **Age**: Raw age value
- **Gender**: Label-encoded (male=0, female=1, other=2)
- **BMI**: Raw BMI value
- **Kids**: Number of children
- **Smoker**: Boolean (0 or 1)
- **Location**: Label-encoded (NY, CA, TX, FL, IL, PA)

### Model Performance
- **Algorithm**: Random Forest (100 trees, max_depth=10)
- **Train R²**: ~0.9858 (98.6% variance explained)
- **Test R²**: ~0.9034 (90.3% variance on unseen data)

## How It Works

1. **User fills the form** with their health/demographic data
2. **Frontend sends JSON POST** to `/api/predict`
3. **Controller validates** input fields
4. **MLPredictionService** spawns a Python subprocess
5. **Python script** loads the trained model and encoders
6. **Prediction is made** using the Random Forest model
7. **Suggestions are generated** based on user profile
8. **Response is returned** to the frontend and displayed

## Advanced Features (Optional)

### Retrain the Model
To train with new data or different hyperparameters:

```powershell
# Edit train_model.py to customize:
# - Dataset generation (size, ranges)
# - Model hyperparameters (n_estimators, max_depth)
# - Feature engineering logic

# Then retrain:
& ".\.venv\Scripts\python.exe" train_model.py
```

### Add Real Insurance Data
Replace the synthetic data generation in `train_model.py` with real data:
```python
# Instead of:
data = { 'age': np.random.randint(...), ... }

# Load from CSV:
df = pd.read_csv('real_insurance_data.csv')
```

### Use Gemini for Suggestions (Optional)
The project still includes `GeminiService` and `Home` controller. You can:
1. Keep the ML model for cost prediction
2. Use Gemini API for enhanced suggestions

Edit `PredictController.predict()` to call both services.

### Export Model to ONNX
For better interoperability, export the scikit-learn model to ONNX:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 6]))]
onyx_model = convert_sklearn(model, initial_types=initial_type)
# Save and use with ONNX Runtime in Java
```

## Troubleshooting

### "Python script error" or "Model files not found"
- Ensure `train_model.py` has been run successfully
- Check that model files exist in `src/main/resources/models/`
- Verify Python path in `MLPredictionService.getPythonExecutable()`

### "Unknown categorical value" (e.g., location not in training set)
- The model only knows locations from training: NY, CA, TX, FL, IL, PA
- Add other locations to `train_model.py` before retraining

### Java cannot find Python executable
- Edit `MLPredictionService.getPythonExecutable()` with your Python path
- Or ensure Python is in system PATH

### Port 8080 already in use
```powershell
# Change in application.properties:
server.port=8081
```

## Example Usage

### From PowerShell:
```powershell
$payload = @{
    name = "Jane Smith"
    age = 35
    gender = "female"
    smoker = $false
    location = "CA"
    kids = 1
    bmi = 25.0
    existingCondition = "none"
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://localhost:8080/api/predict' `
    -Method Post `
    -ContentType 'application/json' `
    -Body $payload
```

### From Frontend:
1. Open http://localhost:8080/
2. Fill the form
3. Click "Get Estimate"
4. View the predicted cost and suggestions

## Security Notes

- **API Key**: If using Gemini, store `gemini.api.key` in environment variables, not in source code
- **Input Validation**: The controller validates required fields and ranges
- **HTTPS**: In production, deploy behind HTTPS
- **Rate Limiting**: Consider adding rate limiting for the `/api/predict` endpoint

## Future Enhancements

- [ ] Add CSV upload for batch predictions
- [ ] Dashboard with historical predictions
- [ ] Model versioning and A/B testing
- [ ] Advanced feature engineering (interactions, polynomials)
- [ ] Hyperparameter tuning with grid search
- [ ] Model explainability (SHAP values)
- [ ] WebSocket for real-time predictions
- [ ] Database persistence for audit trail

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review model training logs in console output
3. Verify Python and Java versions
4. Ensure all dependencies are installed
