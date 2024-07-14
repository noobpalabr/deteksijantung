from flask import Flask, request, jsonify, render_template, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model = joblib.load("heartmodel.joblib")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.form.to_dict()
    age = int(data['age'])
    chest_pain_type = data['chest_pain_type']
    max_heart_rate_achieved = int(data['max_heart_rate_achieved'])
    st_depression = float(data['st_depression'])
    thalassemia = data['thalassemia']
    resting_blood_pressure = int(data['resting_blood_pressure'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame(data=[[age, 1, chest_pain_type, resting_blood_pressure, 0, 0, 0, max_heart_rate_achieved, 0, st_depression, 0, 0, thalassemia]],
                              columns=['Age', 'Sex', 'ChestPainType', 'RestingBloodPressure', 'Cholesterol', 'FatingBloodSugar', 'RestingEcg', 'MaxHeartRateAchieved', 'ExerciseInducedAngina', 'StDepression', 'StSlope', 'NumMajorVessels', 'Thalassemia'])

    all_features = ['Age', 'Sex', 'ChestPainType', 'RestingBloodPressure', 'Cholesterol', 'FatingBloodSugar', 'RestingEcg',
                    'MaxHeartRateAchieved', 'ExerciseInducedAngina', 'StDepression', 'StSlope', 'NumMajorVessels', 'Thalassemia']
    default_values = {'Sex': 1, 'RestingBloodPressure': 0, 'Cholesterol': 0, 'FatingBloodSugar': 0,
                      'RestingEcg': 0, 'ExerciseInducedAngina': 0, 'StSlope': 0, 'NumMajorVessels': 0}

    for feature in all_features:
        if feature not in input_data.columns:
            input_data[feature] = default_values.get(feature, 0)

    input_data = input_data[all_features]

    # Predict using the model
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Prepare the response
    response = {
        'prediction': prediction,
        'probability': prediction_proba.tolist()
    }

    return jsonify(response)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
