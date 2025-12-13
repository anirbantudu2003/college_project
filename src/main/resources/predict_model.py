"""
Python script that loads the trained insurance model and makes predictions.
Called from Java via subprocess with command-line arguments.
Dynamically loads feature names and encoders from model config.

Usage:
  python predict_model.py <age> <gender> <bmi> <kids> <smoker> <location>
"""

import sys
import joblib
import numpy as np
import os
import json

def main():
    if len(sys.argv) < 2:
        print("Error: Expected at least 1 argument", file=sys.stderr)
        sys.exit(1)

    model_dir = 'src/main/resources/models'
    
    try:
        # Load model configuration
        config_path = os.path.join(model_dir, 'model_config.json')
        if not os.path.exists(config_path):
            print("Error: Model config not found. Train a model first.", file=sys.stderr)
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        feature_names = config.get('feature_names', [])
        encoders_list = config.get('encoders', [])
        
        # Load model
        model = joblib.load(os.path.join(model_dir, 'insurance_model.pkl'))
        
        # Load encoders
        encoders = {}
        for encoder_name in encoders_list:
            encoder_path = os.path.join(model_dir, f'label_encoder_{encoder_name}.pkl')
            if os.path.exists(encoder_path):
                encoders[encoder_name] = joblib.load(encoder_path)
        
        # Parse input arguments into a dict
        # Format: key1=value1 key2=value2 ... or positional args
        input_dict = {}
        if len(sys.argv) > 1 and '=' in sys.argv[1]:
            # JSON or key=value format
            for arg in sys.argv[1:]:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    # Normalize boolean values
                    if value.lower() in ('true', 'false'):
                        input_dict[key] = 'yes' if value.lower() == 'true' else 'no'
                    else:
                        input_dict[key] = value
        else:
            # Positional arguments (legacy format)
            if len(sys.argv) != 7:
                print("Error: Expected 6 positional arguments or key=value pairs", file=sys.stderr)
                sys.exit(1)
            smoker_val = sys.argv[5].lower()
            if smoker_val in ('true', '1', 'yes'):
                smoker_val = 'yes'
            elif smoker_val in ('false', '0', 'no'):
                smoker_val = 'no'
            input_dict = {
                'age': sys.argv[1],
                'gender': sys.argv[2],
                'bmi': sys.argv[3],
                'children': sys.argv[4],
                'smoker': smoker_val,
                'region': sys.argv[6]
            }
        
        # Build feature vector based on config
        features = []
        for feature in feature_names:
            if feature.endswith('_encoded'):
                # Categorical feature - find the original column
                col_name = feature.replace('_encoded', '')
                if col_name in encoders:
                    # Try multiple possible input keys
                    value = None
                    if col_name in input_dict:
                        value = input_dict[col_name]
                    elif col_name + 's' in input_dict:  # plural form
                        value = input_dict[col_name + 's']
                    elif col_name == 'children' and 'kids' in input_dict:
                        value = input_dict['kids']
                    elif col_name == 'region' and 'location' in input_dict:
                        value = input_dict['location']
                    
                    if value is None:
                        print(f"Error: Missing value for column '{col_name}'. Available keys: {list(input_dict.keys())}", file=sys.stderr)
                        sys.exit(1)
                    
                    try:
                        encoded_value = encoders[col_name].transform([str(value).lower()])[0]
                        features.append(encoded_value)
                    except ValueError as ve:
                        print(f"Error: Unknown value '{value}' for column '{col_name}'. Valid values: {list(encoders[col_name].classes_)}", file=sys.stderr)
                        sys.exit(1)
            else:
                # Numeric feature
                value = None
                if feature in input_dict:
                    value = input_dict[feature]
                elif feature == 'children' and 'kids' in input_dict:
                    value = input_dict['kids']
                
                if value is not None:
                    try:
                        features.append(float(value))
                    except ValueError:
                        print(f"Error: Invalid numeric value for '{feature}': {value}", file=sys.stderr)
                        sys.exit(1)
        
        
        # Make prediction
        if not features:
            print("Error: No features provided", file=sys.stderr)
            sys.exit(1)
        
        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        
        # Output only the numeric prediction (so Java can easily parse it)
        print(f"{prediction:.2f}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
