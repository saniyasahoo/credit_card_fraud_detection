# credit_card_fraud_detection
A machine learning–based web application that detects fraudulent credit card transactions using a trained classification model and a Flask web interface.

Features:
Machine Learning model for fraud detection
Flask-based web application
CSV file upload for transaction data
Predicts fraudulent vs legitimate transactions
Simple and user-friendly web UI

Tech Stack:
Programming Language: Python
Web Framework: Flask
Machine Learning: Scikit-learn
Data Processing: Pandas, NumPy
Model Saving: Joblib
Frontend: HTML, CSS

Dataset:
Uses the Credit Card Fraud Detection Dataset (from Kaggle)
Dataset is not included in this repository due to large file size
You can download it from Kaggle and place it locally

How to Run the Project:
pip install -r requirements.txt
python train_model.py
python app.py
(Open in browser)
http://127.0.0.1:5000/

Machine Learning Model
Algorithm: Classification model (e.g., Random Forest)
Trained to distinguish between fraudulent and non-fraudulent transactions
