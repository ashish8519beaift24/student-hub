from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
import os

app = Flask(__name__)

study_model_path = 'study_model.pkl'

if os.path.exists(study_model_path):
    study_model = joblib.load(study_model_path)
else:
    study_model = None
    print("ERROR: study_model.pkl file not found! Please run train_study_model.py")

# Home Route (Redirect to Study Predictor)
@app.route('/')
def home():
    return redirect(url_for('study_predict'))

# GPA Calculator Route
@app.route('/gpa', methods=['GET', 'POST'])
def gpa_calculator():
    gpa_result = None
    if request.method == 'POST':
        try:
            # We'll receive lists of grades and credits
            grades = request.form.getlist('grade')
            credits = request.form.getlist('credits')
            
            total_points = 0
            total_credits = 0
            
            grade_scale = {
                'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'F': 0.0
            }
            
            for grade, credit in zip(grades, credits):
                if grade in grade_scale and credit.strip() != '':
                    c = float(credit)
                    total_points += grade_scale[grade] * c
                    total_credits += c
                    
            if total_credits > 0:
                gpa_result = round(total_points / total_credits, 2)
            else:
                gpa_result = 0.0
                
        except Exception as e:
            gpa_result = f"Error: {str(e)}"
            
    return render_template('gpa.html', gpa_result=gpa_result)

# Study Score Predictor Route
@app.route('/study_predict', methods=['GET', 'POST'])
def study_predict():
    prediction_text = None
    if request.method == 'POST':
        try:
            hours = float(request.form['hours'])
            attendance = float(request.form['attendance'])
            
            if study_model:
                data = np.array([[hours, attendance]])
                prediction = study_model.predict(data)
                result = round(prediction[0], 2)
                # Cap at 100
                if result > 100: result = 100
                prediction_text = f'Predicted Exam Score: {result}%'
            else:
                prediction_text = 'Error: Study model not found.'
                
        except Exception as e:
            prediction_text = f'Error: {str(e)}'
            
    return render_template('study_predictor.html', prediction_text=prediction_text)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, port=5001)