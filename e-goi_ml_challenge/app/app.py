#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved model, encoder, and scaler
model = joblib.load('../models/random_forest_model.pkl')
encoder = joblib.load('../models/encoder.pkl')
scaler = joblib.load('../models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()
        
        # Convert the JSON data into a DataFrame
        input_data = pd.DataFrame([data])
        
        # One-hot encode categorical columns
        categorical_cols = input_data.select_dtypes(include='object').columns.tolist()
        encoded_data = encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

        # Normalize numerical columns
        numerical_cols = input_data.select_dtypes(include='int64').columns.tolist()
        scaled_data = scaler.transform(input_data[numerical_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)

        # Combine both processed data
        processed_data = pd.concat([scaled_df, encoded_df], axis=1)

        # Make the prediction using the Random Forest model
        prediction = model.predict(processed_data)

        # Return the prediction in the response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
