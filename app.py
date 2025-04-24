from flask import Flask, send_file, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return send_file('index.html')
 
@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join('static', 'fevicon.png'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform(pd.DataFrame([features]))
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        result = "Malignant" if prediction == 0 else "Benign"
        labels = ['Malignant', 'Benign']
        
        return render_template(
            'index.html',
            prediction_text=result,  # only 'Malignant' or 'Benign'
            prediction_result=f'Prediction: {result}',  # for display if needed
            labels=labels,
            probabilities=probabilities.tolist(),
            field_values=request.form  # send input values back
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
