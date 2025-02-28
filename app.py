from flask import Flask, render_template, request
import numpy as np
import pandas as pd  # Import pandas to handle DataFrame
import joblib
import os

app = Flask(__name__)
filename = 'file_model.pkl'
model = joblib.load(filename)

# Mapping predictions to labels
prediction_labels = {0: "Healthy", 1: "Sleep Apnea", 2: "Insomnia"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        Age = float(request.form['age'])
        Sleep_Duration = float(request.form['sleep_duration'])
        Quality_of_Sleep = float(request.form['quality_of_sleep'])
        Occupation_Salesperson = int(request.form['occupation_salesperson'])
        BMICategory_Overweight = int(request.form['category_overweight'])
        Stress_Level = float(request.form['stress_level'])
        Occupation_Doctor = int(request.form['occupation_doctor'])

        # Define the feature names used during training (must match exactly)
        feature_names = [
            "Age", "Sleep Duration", "Quality of Sleep",
            "Occupation_Salesperson", "BMI Category_Overweight",
            "Stress Level", "Occupation_Doctor"
        ]

        # Convert the input data into a DataFrame
        features_df = pd.DataFrame([[Age, Sleep_Duration, Quality_of_Sleep,
                                     Occupation_Salesperson, BMICategory_Overweight,
                                     Stress_Level, Occupation_Doctor]],
                                   columns=[
                                       "Age", "Sleep_Duration", "Quality_of_Sleep",
                                       "Occupation_Salesperson", "BMICategory_Overweight",
                                       "Stress_Level", "Occupation_Doctor"
                                   ])

        # Rename columns to match the feature names used during training
        features_df.rename(columns={
            "Sleep_Duration": "Sleep Duration",
            "Quality_of_Sleep": "Quality of Sleep",
            "BMICategory_Overweight": "BMI Category_Overweight",
            "Stress_Level": "Stress Level"
        }, inplace=True)

        # Debugging: Print feature names before prediction
        print("Feature columns before prediction:", features_df.columns)

        # Model prediction
        pred = model.predict(features_df)[0]
        pred_label = prediction_labels.get(pred, "Unknown")  # Convert numeric prediction to label

        return render_template('index.html', predict=pred_label)

    except Exception as e:
        print("Error:", e)  # Print error for debugging
        return render_template('index.html', predict="Error processing request")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)
