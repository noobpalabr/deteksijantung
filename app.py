from flask import Flask, request, jsonify, render_template, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model = joblib.load("heartmodel.joblib")

# Function to map categorical values to numerical values
def map_category_values(data):
    mapping = {
        'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
        'Thalassemia': {'normal': 1, 'fixed defect': 2, 'reversable defect': 3}
    }
    for column, map_dict in mapping.items():
        data[column] = data[column].map(map_dict)
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.form.to_dict()
    age = int(data['age'])
    chest_pain_type = data['cp']
    max_heart_rate_achieved = int(data['thalach'])
    st_depression = float(data['oldpeak'])
    thalassemia = data['thal']

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'age': [age],
        'ChestPainType': [chest_pain_type],
        'thalach': [max_heart_rate_achieved],
        'oldpeak': [st_depression],
        'thal': [thalassemia]
    })

    # Map categorical values to numerical values
    input_data = map_category_values(input_data)

    # Define all features and default values
    all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope']
    default_values = {
        'sex': 1,
        'trestbps': 120,  # Example default value, replace with mean or median if available
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

    # Predict using the model
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Prepare the response
    response = {
        'prediction': 'Healthy' if prediction == 0 else 'Heart Disease',
        'probability': prediction_proba.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
