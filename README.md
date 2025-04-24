# 🧬 Breast Cancer Prediction Web App

A machine learning-powered web application built with Flask for predicting whether a breast tumor is **benign** or **malignant** using diagnostic measurements from the Wisconsin Breast Cancer Dataset. 

This intelligent assistant helps support early diagnosis through advanced ML models and intuitive visual feedback.

---

## 🚀 Features

- 🔮 **Predict Tumor Type** — Instant classification into *Benign* or *Malignant*
- 📊 **Visual Confidence Score** — Dynamic Pie Chart for prediction probabilities
- 🧠 **Feature Importance Analysis** — Doughnut Chart showing key diagnostic features
- 🔁 **Form Input Retention** — Automatically keeps user input after submission
- 💬 **Smart Result Alerts** — Critical alerts for malignant results with recommendations
- 💡 **Clean UI** — Elegant and accessible frontend layout

---

## 🖼 Screenshots

### 🏠 Homepage  
> Clean UI, all diagnostic input fields, and system introduction

![SS1](https://github.com/user-attachments/assets/86fffe53-7424-47aa-bbc4-35435c921724)

---

### ❌ Cancer Detected (Malignant Result)  
> Red alert with doctor recommendation

![SS2](https://github.com/user-attachments/assets/49ab0ab7-c612-4259-b257-d62d66391d71)

---

### ✅ Cancer Not Detected (Benign Result)  
> Green confirmation with follow-up advice

![SS3](https://github.com/user-attachments/assets/a3dfd05c-9643-405b-804e-5a40358cb9c2)

---

## 🧰 Tech Stack

- Python 3.x
- Flask
- Scikit-learn (SVM Classifier)
- MinMaxScaler
- Chart.js (for visualizations)
- HTML + CSS + Jinja2 Templates

---

## 🛠 Installation & Setup

1. **Clone the repository**

      git clone https://github.com/yourusername/breast-cancer-prediction-app.git
      
      cd breast-cancer-prediction-app

2. **Install the dependencies**

      pip install -r requirements.txt

3. **Run the app**

      python app.py

4. **Visit in Browser**

      http://127.0.0.1:5000
