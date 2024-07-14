from flask import Flask, request, jsonify, render_template, url_for
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = joblib.load("heartmodel.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        print("Received form data:", data)
        
        age = int(data.get('age'))
        cp = int(data.get('chest_pain_type'))
        thalach = int(data.get('max_heart_rate_achieved'))
        oldpeak = float(data.get('st_depression'))
        thal = int(data.get('thalassemia'))
        trestbps = int(data.get('resting_blood_pressure'))

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'age': [age],
            'cp': [cp],
            'thalach': [thalach],
            'oldpeak': [oldpeak],
            'thal': [thal],
            'trestbps': [trestbps]
        })

        # Print input data before adding default values
        print("Input data before adding default values:", input_data)

        # Define all features and default values
        all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope']
        default_values = {
            'sex': 1,
            'chol': 200,      # Example default value, replace with mean or median if available
            'fbs': 0,
            'restecg': 0,
            'exang': 0,
            'slope': 0
        }

        # Add missing features with default values
        for feature in all_features:
            if feature not in input_data.columns:
                input_data[feature] = default_values.get(feature, 0)

        # Ensure the input data has the correct order of features
        input_data = input_data[all_features]

        # Print input data before prediction
        print("Input data before prediction:", input_data)

        # Check for NaN, infinity or excessively large values
        if input_data.isnull().values.any():
            raise ValueError("Input contains NaN values.")
        if np.isinf(input_data.values).any():
            raise ValueError("Input contains infinity values.")
        if (np.abs(input_data.values) > np.finfo(np.float32).max).any():
            raise ValueError("Input contains values too large for dtype('float32').")

        # Predict using the model
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Prepare the response
        result = 'Healthy' if prediction == 0 else 'Heart Disease'
        probability = prediction_proba.tolist()

        return render_template('index.html', result=result, probability=probability)
    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
