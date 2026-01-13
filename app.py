from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("fraud_model.joblib")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    try:
        df = pd.read_csv(file)

        # Drop unnecessary columns if present
        df.drop(columns=[col for col in ['Time', 'Class'] if col in df.columns], inplace=True)
        
        # Scale the 'Amount' column
        df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])

        # Reorder columns to match the model's expected features
        expected_features = ['V' + str(i) for i in range(1, 29)] + ['Amount']
        df = df[expected_features]

        # Make predictions
        probs = model.predict_proba(df)
        predictions = (probs[:, 1] > 0.5).astype(int)
        df['Prediction'] = predictions
        df['Confidence'] = probs[:, 1].round(4)

        # Summary
        total = len(predictions)
        fraud = int((predictions == 1).sum())
        legit = total - fraud

        # Top 10 fraud predictions
        fraud_rows = df[df['Prediction'] == 1].head(10).to_dict(orient='records')

        return render_template(
            "index.html",
            total=total,
            fraud=fraud,
            legit=legit,
            fraud_rows=fraud_rows
        )

    except Exception as e:
        return f"Error processing file: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)