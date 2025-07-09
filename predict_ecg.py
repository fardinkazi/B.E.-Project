import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model
import joblib

# Load the saved model and scaler
model = load_model("ecg_cnn_model.h5")
scaler = joblib.load("scaler.save")

# Load new ECG input
input_file = "ecg_input.csv"
input_data = pd.read_csv(input_file, header=None)

# Scale and reshape input
X_input = scaler.transform(input_data.values)
X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)

# Predict
predictions = model.predict(X_input)
binary_predictions = (predictions > 0.5).astype(int)

# Output prediction
for i, pred in enumerate(binary_predictions):
    print(f"Sample {i+1} prediction: {'Disease Detected' if pred == 1 else 'Normal'}")
