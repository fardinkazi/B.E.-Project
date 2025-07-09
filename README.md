# 🩺 Smart ECG Classification using Deep Learning

## 📌 Project Overview

This project is a **Smart ECG Monitoring System** that collects ECG signals using a hardware setup built with the **AD8232 ECG sensor** and **ESP8266 (NodeMCU)**. The signals are then sent to a Python-based system, where they are **preprocessed** and passed to a **deep learning model** to classify them as **normal** or **abnormal**.

The project aims to provide **cost-effective early detection** of heart conditions and enable **remote health monitoring**.

---

## 🔧 Tech Stack

- **Programming Language:** Python
- **Microcontroller Code:** Embedded C / Arduino IDE
- **Deep Learning Framework:** TensorFlow / Keras
- **Data Visualization:** NumPy, Matplotlib
- **Frontend (Optional):** HTML, CSS
- **Backend (Optional):** Flask (for web dashboard)
- **Communication:** Serial UART / Wi-Fi (ESP8266)

---

## 🔬 Hardware Components

- 🧠 **AD8232 ECG Sensor**
- 📡 **ESP8266 NodeMCU**
- 🔌 USB Cable & Power Supply
- 🧷 Jumper Wires & Breadboard

---

## 🧠 Machine Learning Model

- **Model Type:** 1D Convolutional Neural Network (CNN)
- **Training Data:** MIT-BIH Arrhythmia Dataset / Self-Collected ECG Data
- **Preprocessing:** Filtering, R-R Interval Detection, Normalization
- **Model Accuracy:** ~`XX%` (replace with actual)
- **Output Classes:** Normal, Abnormal

---

<details>
<summary><strong>🔁 System Workflow Diagram (Click to expand)</strong></summary>

```mermaid
graph TD
    A[ECG Sensor (AD8232)] --> B[ESP8266 Microcontroller]
    B --> C[Serial Communication]
    C --> D[Python Data Receiver]
    D --> E[Preprocessing]
    E --> F[Deep Learning Model]
    F --> G[Prediction: Normal/Abnormal]
    G --> H[Display / Store in Dashboard]