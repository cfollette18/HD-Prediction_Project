from flask import Flask, request, jsonify
from flask_cors import CORS  
import shap
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Create Flask app
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # For testing purposes, use dummy data
        dummy_data = {
            'age': 50,
            'thalach': 150,
            'oldpeak': 1.5
        }
        input_data = pd.DataFrame([dummy_data])
    else:
        # Handle POST request
        data = request.json
        input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Compute SHAP values
    shap_values = explainer.shap_values(input_data)

    # Prepare response
    response = {
        'prediction': int(prediction),
        'shap_values': shap_values[0].tolist(),
        'feature_names': input_data.columns.tolist(),
        'feature_values': input_data.values[0].tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)