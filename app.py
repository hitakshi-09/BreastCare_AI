from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        features = [float(input_data[field]) for field in input_data]
        scaled_features = scaler.transform(pd.DataFrame([features]))

        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        result = "Malignant" if prediction == 0 else "Benign"
        labels = ['Malignant', 'Benign']

        feature_labels = [
            'Worst Radius', 'Mean Radius', 'Worst Perimeter', 'Mean Perimeter',
            'Worst Area', 'Mean Area', 'Worst Concave Points', 'Mean Concave Points',
            'Worst Compactness', 'Mean Texture'
        ]

        return render_template(
            'index.html',
            prediction_text=result,
            labels=labels,
            probabilities=probabilities.tolist(),
            feature_labels=feature_labels,
            features=features[:10],  # First 10 feature values for doughnut chart
            input_data=input_data
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', input_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
