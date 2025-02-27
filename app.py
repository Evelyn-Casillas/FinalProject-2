from flask import Flask, render_template, request
import numpy as np
import joblib
import os  # Added this line

app = Flask(__name__)
filename = 'file_model.pkl'

# Load the model (adjust the model to accept 11 features)
model = joblib.load(filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    Age = float(request.form['age'])
    Sleep_Duration = float(request.form['sleep_duration'])
    Quality_of_Sleep = float(request.form['quality_of_sleep'])
    Occupation_Salesperson = int(request.form['occupation_salesperson'])
    BMICategory_Overweight = int(request.form['category_overweight'])
    Stress_Level = float(request.form['stress_level'])
    Occupation_Doctor = int(request.form['occupation_doctor'])

    features = np.array([[Age, Sleep_Duration, Quality_of_Sleep, Occupation_Salesperson,
                          BMICategory_Overweight, Stress_Level, Occupation_Doctor]])

    # Predict using the model
    pred = model.predict(features)

    return render_template('index.html', predict=str(pred))

# Modified for Heroku deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002)) #Use 5001 or 5000 for local testing. Heroku will override.
    app.run(host='0.0.0.0', port=port, debug=False) #Debug set to false.