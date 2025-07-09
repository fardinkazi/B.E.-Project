from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load model and scaler
model = load_model("ecg_cnn_model.h5")
scaler = joblib.load("scaler.save")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = None
    waveform_image = None
    user_data = None

    if request.method == "POST":
        # Fetch user info
        name = request.form.get("name")
        age = request.form.get("age")
        sex = request.form.get("sex")
        email = request.form.get("email")
        symptoms = request.form.get("symptoms")

        user_data = {
            "Name": name,
            "Age": age,
            "Sex": sex,
            "Email": email,
            "Symptoms": symptoms
        }

        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            try:
                if filepath.endswith(".csv"):
                    input_data = pd.read_csv(filepath, header=None)
                elif filepath.endswith(".xls") or filepath.endswith(".xlsx"):
                    input_data = pd.read_excel(filepath, header=None)
                else:
                    return render_template("index.html", error="Unsupported file type")

                # Save ECG plot
                plt.figure(figsize=(10, 4))
                plt.plot(input_data.iloc[0].values)
                plt.title("ECG Waveform (Sample 1)")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                waveform_path = os.path.join(STATIC_FOLDER, "waveform.png")
                plt.tight_layout()
                plt.savefig(waveform_path)
                plt.close()
                waveform_image = "static/waveform.png"

                # Predict
                X_input = scaler.transform(input_data.values)
                X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)
                predictions = model.predict(X_input)
                binary_predictions = (predictions > 0.5).astype(int)

                prediction_results = [
                    f"Sample {i+1}: {'Disease Detected' if pred == 1 else 'Normal'}"
                    for i, pred in enumerate(binary_predictions)
                ]

            except Exception as e:
                return render_template("index.html", error=f"Prediction error: {str(e)}")

    return render_template("index.html", predictions=prediction_results, waveform=waveform_image, user=user_data)

if __name__ == "__main__":
    app.run(debug=True)
