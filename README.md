# 🩺 Smart ECG Classification using Deep Learning



## 📌 Project Overview

This project is a **Smart ECG Monitoring System** designed to collect ECG signals using a hardware setup based on the **AD8232 ECG module** and **ESP8266 (NodeMCU)**. The signals are then preprocessed and passed to a deep learning model for **real-time classification** into **normal** or **abnormal** heart activity.

This solution is ideal for early detection of heart anomalies and remote health monitoring.


## 🔧 Tech Stack

- **Programming Language:** Python
- **Embedded C** for microcontroller firmware
- **Machine Learning Framework:** TensorFlow / Keras
- **Data Visualization:** Matplotlib, NumPy
- **Frontend:** HTML/CSS (for dashboard, optional)
- **Backend (optional):** Flask (for web-based implementation)
- **Communication:** Serial (UART), Wi-Fi (ESP8266)


## 🔬 Hardware Components

- 🧠 **AD8232 ECG Sensor Module**
- 📡 **ESP8266 NodeMCU Wi-Fi Module**
- ⚙️ **Jumper Wires**
- 🔌 **Power Source / USB Cable**

---

## 🧠 Machine Learning Model

- **Model Type:** 1D Convolutional Neural Network (CNN)
- **Dataset Used:** MIT-BIH Arrhythmia Dataset / Self-collected ECG via AD8232
- **Preprocessing:** Denoising, Normalization, R-R Interval Detection
- **Accuracy Achieved:** ~`XX%` (replace with your final value)
- **Output Classes:** Normal, Abnormal

---

## 🔁 System Workflow

```mermaid
graph TD;
    A[ECG Sensor (AD8232)] --> B[ESP8266 Microcontroller];
    B --> C[Serial Communication];
    C --> D[Python Data Receiver];
    D --> E[Preprocessing];
    E --> F[Deep Learning Model];
    F --> G[Prediction: Normal/Abnormal];
    G --> H[Display / Store in Dashboard];