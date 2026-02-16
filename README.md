# 🫀 Real-Time ECG Heartbeat Classification System

A complete **end-to-end machine learning pipeline** for real-time ECG heartbeat classification using signal processing, feature engineering, and supervised learning.  
This project simulates **live ECG monitoring** and performs **real-time predictions with visualization**.

> 🚀 Portfolio Highlight: Demonstrates data preprocessing → feature extraction → model training → live inference → real-time visualization.

---

## 📌 Features

- 📈 Real-time ECG signal visualization
- 🧠 Machine Learning–based heartbeat classification
- ⚡ Live prediction with confidence scores
- 📊 Probability distribution across heartbeat classes
- 🧪 Robust feature alignment for production inference
- 🔁 Continuous streaming simulation

---

## 🧠 Heartbeat Classes

| Label | Class Name |
|------|-----------|
| 0 | Normal |
| 1 | Supraventricular |
| 2 | Premature Ventricular |
| 3 | Fusion |
| 4 | Unclassifiable |

---

---

## 🔬 Feature Engineering

### Time Domain
- Mean, Standard Deviation
- Min, Max, Range
- RMS
- Peak count & peak statistics

### Frequency Domain
- Spectral energy
- Spectral centroid
- Band energy ratios (Low / Mid / High)

### Statistical
- Skewness
- Kurtosis
- Zero-crossing rate

### Wavelet Domain
- Discrete Wavelet Transform (DB4)
- Energy, mean, and standard deviation of coefficients

---

## 🧠 Model Pipeline

1. ECG signal preprocessing  
2. Feature extraction (time + frequency + wavelet)
3. Feature scaling using StandardScaler
4. Supervised ML classification
5. Probability-based confidence estimation
6. Real-time visualization using Matplotlib animation

---

## ▶️ Real-Time Demo

Run the live prediction demo:

```bash
python predict_live.py



## 🏗️ Project Architecture

