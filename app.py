from flask import Flask, render_template, request
import numpy as np
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
    # Get input data
    Age = float(request.form['age'])
    Sleep_Duration = float(request.form['sleep_duration'])
    Quality_of_Sleep = float(request.form['quality_of_sleep'])
    Occupation_Salesperson = int(request.form['occupation_salesperson'])
    BMICategory_Overweight = int(request.form['category_overweight'])
    Stress_Level = float(request.form['stress_level'])
    Occupation_Doctor = int(request.form['occupation_doctor'])

    features = np.array([[Age, Sleep_Duration, Quality_of_Sleep, Occupation_Salesperson,
                          BMICategory_Overweight, Stress_Level, Occupation_Doctor]])

    # Model prediction
    pred = model.predict(features)[0]
    pred_label = prediction_labels.get(pred, "Unknown")  # Convert to label

    return render_template('index.html', predict=pred_label)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)
