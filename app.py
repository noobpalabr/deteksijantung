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
        'cp': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
        'thal': {'normal': 1, 'fixed defect': 2, 'reversable defect': 3}
    }
    for column, map_dict in mapping.items():
        data[column] = data[column].map(map_dict)
    return data

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
        cp = data.get('chest_pain_type')
        thalach = int(data.get('max_heart_rate_achieved'))
        oldpeak = float(data.get('st_depression'))
        thal = data.get('thalassemia')
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

        # Map categorical values to numerical values
        input_data = map_category_values(input_data)

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

        # Predict using the model
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Prepare the response
        response = {
            'prediction': 'Healthy' if prediction == 0 else 'Heart Disease',
            'probability': prediction_proba.tolist()
        }

        return jsonify(response)
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
