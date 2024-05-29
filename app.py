import os
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# import json

# Load models
models = {
    'math': {
        'Support Vector Regression': joblib.load('models/support_vector_regression_math_model.pkl'),
        'Linear Regression': joblib.load('models/linear_regression_math_model.pkl'),
        'Gradient Boosting': joblib.load('models/gradient_boosting_math_model.pkl'),
        'Decision Tree': joblib.load('models/decision_tree_math_model.pkl')
    },
    'reading': {
        'Support Vector Regression': joblib.load('models/support_vector_regression_reading_model.pkl'),
        'Linear Regression': joblib.load('models/linear_regression_reading_model.pkl'),
        'Gradient Boosting': joblib.load('models/gradient_boosting_reading_model.pkl'),
        'Decision Tree': joblib.load('models/decision_tree_reading_model.pkl')
    },
    'writing': {
        'Support Vector Regression': joblib.load('models/support_vector_regression_writing_model.pkl'),
        'Linear Regression': joblib.load('models/linear_regression_writing_model.pkl'),
        'Gradient Boosting': joblib.load('models/gradient_boosting_writing_model.pkl'),
        'Decision Tree': joblib.load('models/decision_tree_writing_model.pkl')
    }
}

# Load encoder and scaler
encoder = joblib.load('models/label_encoder.pkl')
scaler = joblib.load('models/standard_scaler.pkl')

# Load model accuracies
with open('models/model_accuracies.json', 'r') as f:
    model_accuracies = json.load(f)

# Define the form route
@app.route('/')
def form():
    return render_template('form.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gender = request.form['gender']
    race_ethnicity = request.form['race_ethnicity']
    parental_level_of_education = request.form['parental_level_of_education']
    lunch = request.form['lunch']
    test_preparation_course = request.form['test_preparation_course']

    # Create DataFrame for input
    input_data = pd.DataFrame([[gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course]],
                              columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])

    # Encode categorical data
    input_encoded = pd.DataFrame(encoder.transform(input_data).toarray(),
                                 columns=encoder.get_feature_names_out())

    # Get column names from input_encoded
    input_encoded_columns = input_encoded.columns

    # Ensure consistency in feature names
    # Add missing columns with zeros if necessary
    missing_columns = set(scaler.mean_) - set(input_encoded_columns)
    for col in missing_columns:
        input_encoded[col] = 0

    # Reorder columns to match the scaler's feature order
    input_encoded = input_encoded[scaler.mean_]

    # Scale input data
    input_scaled = scaler.transform(input_encoded)

    # Predict scores and include model accuracies
    predictions = {}
    for subject, subject_models in models.items():
        predictions[subject] = {}
        for model_name, model in subject_models.items():
            prediction = model.predict(input_scaled.reshape(1, -1))
            accuracy = model_accuracies[subject][model_name] * 100 
            predictions[subject][model_name] = {
                'score': int(prediction[0]),
                'accuracy': accuracy
            }

    return render_template('results.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
