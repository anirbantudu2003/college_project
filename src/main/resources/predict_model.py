"""
Python script that loads the trained insurance model and makes predictions.
Called from Java via subprocess with command-line arguments.
Dynamically loads feature names and encoders from model config.

Usage:
  python predict_model.py <age> <gender> <bmi> <kids> <smoker> <location> [--model <model_name>]
  
Model options: random_forest, decision_tree, linear_regression
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
    selected_model = 'random_forest'  # Default model
    
    # Check if --model argument is provided
    args = sys.argv[1:]
    if '--model' in args:
        model_idx = args.index('--model')
        if model_idx + 1 < len(args):
            selected_model = args[model_idx + 1]
            # Remove --model and its value from args
            args = args[:model_idx] + args[model_idx + 2:]
            sys.argv = [sys.argv[0]] + args
    
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
        available_models = config.get('models', ['random_forest'])
        
        # Validate selected model
        if selected_model not in available_models:
            print(f"Error: Invalid model '{selected_model}'. Available models: {available_models}", file=sys.stderr)
            sys.exit(1)
        
        # Load the selected model
        model_files = {
            'random_forest': 'insurance_model.pkl',
            'decision_tree': 'insurance_model_decision_tree.pkl',
            'linear_regression': 'insurance_model_linear_regression.pkl'
        }
        
        model_file = model_files.get(selected_model)
        if not model_file:
            print(f"Error: Model file not found for '{selected_model}'", file=sys.stderr)
            sys.exit(1)
        
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' does not exist. Train the model first.", file=sys.stderr)
            sys.exit(1)
        
        model = joblib.load(model_path)
        
        # Load encoders
        encoders = {}
        for encoder_name in encoders_list:
            encoder_path = os.path.join(model_dir, f'label_encoder_{encoder_name}.pkl')
            if os.path.exists(encoder_path):
                encoders[encoder_name] = joblib.load(encoder_path)
        
        # Parse input arguments into a dict
        # Format: key1=value1 key2=value2 ... or positional args
        input_dict = {}
        positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('--') and sys.argv[sys.argv.index(arg) - 1] != '--model']
        
        if len(positional_args) > 0 and '=' in positional_args[0]:
            # JSON or key=value format
            for arg in positional_args:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    # Normalize boolean values
                    if value.lower() in ('true', 'false'):
                        input_dict[key] = 'yes' if value.lower() == 'true' else 'no'
                    else:
                        input_dict[key] = value
        else:
            # Positional arguments (legacy format)
            if len(positional_args) != 6:
                print(f"Error: Expected 6 positional arguments, got {len(positional_args)}. Args: {positional_args}", file=sys.stderr)
                sys.exit(1)
            smoker_val = positional_args[4].lower()
            if smoker_val in ('true', '1', 'yes'):
                smoker_val = 'yes'
            elif smoker_val in ('false', '0', 'no'):
                smoker_val = 'no'
            # Accept location as region (since Python script calls it location in Java)
            location_val = positional_args[5]
            input_dict = {
                'age': positional_args[0],
                'gender': positional_args[1],
                'bmi': positional_args[2],
                'children': positional_args[3],
                'smoker': smoker_val,
                'region': location_val,
                'location': location_val  # Add as both for compatibility
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
                elif feature == 'kids' and 'children' in input_dict:
                    value = input_dict['children']
                elif feature == 'smoker_int' and 'smoker' in input_dict:
                    # Convert yes/no to 1/0
                    smoker_val = input_dict['smoker']
                    value = 1 if smoker_val in ('yes', '1', 'true', True) else 0
                
                if value is not None:
                    try:
                        features.append(float(value))
                    except ValueError:
                        print(f"Error: Invalid numeric value for '{feature}': {value}", file=sys.stderr)
                        sys.exit(1)
                else:
                    print(f"Error: Missing value for numeric feature '{feature}'. Available keys: {list(input_dict.keys())}", file=sys.stderr)
                    sys.exit(1)
        
        # Make prediction
        if not features:
            print("Error: No features provided", file=sys.stderr)
            sys.exit(1)
        
        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        
        # Output prediction with model name in JSON format for easier parsing
        output = {
            'model': selected_model,
            'prediction': float(prediction)
        }
        print(json.dumps(output))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
