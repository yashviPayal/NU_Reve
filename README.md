# Reve Digital Farming: Intelligent Algorithms for Soil Health & Crop Management

![Python](https://shields.io)
![Machine Learning](https://shields.io)
![Hackathon](https://shields.io)

An intelligent algorithm framework built for the **Reve Sponsored Track: HackNUthon 2024**. This project focuses on analyzing soil properties and optimizing crop management by processing spectro-radiometer readings across varying soil moisture levels.

## 👥 Team: CodeCrafter
* **Prakruti Tank**
* **Yashvi Dalsaniya**
* **Renshi Tarapara**
* **Jignal Vasava**

---

## 📌 Project Overview
Varying moisture levels drastically affect soil sensor readings. This project addresses the challenge by preprocessing raw sensor data, categorizing moisture states, performing advanced correlation analysis on core soil properties (pH, Nitrogen, Phosphorus, Potassium) and building robust predictive machine learning pipelines.

---

## 🛠️ System Architecture & Workflow

### 1. Data Flow Pipeline

```text
┌─────────────────┐      ┌─────────────────────────────┐      ┌─────────────────────────┐
│   Raw Dataset   │ ───> │ Data Preprocessing & Clean  │ ───> │ Moisture Stratification │
└─────────────────┘      └─────────────────────────────┘      └─────────────────────────┘
                                                                           │
                                                                           ▼
┌─────────────────┐      ┌─────────────────────────────┐      ┌─────────────────────────┐
│   Predictions   │ <─── │ Model Evaluation (R² Score) │ <─── │   Advanced Model ML     │
└─────────────────┘      └─────────────────────────────┘      └─────────────────────────┘
```

### 2. Core Execution Steps
* **Load Data:** Imports raw agricultural data files (`.csv`).
* **Feature Extraction & Cleaning:** Uses Regex to extract `Sample Name`, `Moisture Level`, and `Reading Number`. Converts metrics to structured integers and drops corrupt rows.
* **Moisture Categorization:** Stratifies data into controlled environments:
  *  **Low** (0ml)
  *  **Medium** (25ml)
  *  **High** (50ml)
* **Statistical Grouping:** Groups datasets by Sample/Moisture tags to compute baseline mathematical means.

---

## 📊 Analytics & Statistical Insights
To understand the relationship between features across different moisture states, the system maps:
* **Dual-Correlation Mapping:** Computes both **Karl Pearson** and **Spearman** correlation matrices for target soil properties.
* **Visual Heatmaps:** Plots side-by-side matrices to assess dynamic alterations in target variables (pH, Nitro, Posh Nitro, Pota Nitro).

---

## 🤖 Machine Learning Framework

The project trains and evaluates four specialized regression algorithms to map optimal wavelengths and predict final soil health parameters:

1. **Gradient Boosting Regressor (GBR):** Isolates highly impactful wavelengths and handles subtle feature transitions.
2. **Random Forest Regressor (RFR):** Reduces noise and stabilizes variance by aggregating ensemble decision trees.
3. **XGBoost Regressor (XGBR):** Standardizes gradient boosting architectures to ensure high feature-importance throughput.
4. **Linear Regression:** Deployed as a baseline model benchmark.

### 📈 Performance Metric
Model accuracy is evaluated strictly using the **R² (Coefficient of Determination) Score**:
* **R² = 1** implies a Perfect Prediction.
* **R² <= 0** implies the model performs no better than predicting the baseline mean.

---

## 🚀 Future Scope & Enhancements

To take this framework from a hackathon prototype to a field-ready production application, the following updates are planned:

* **Real-time IoT Integration:** Connect the pipeline directly to field-deployed spectro-radiometer hardware sensors via MQTT protocols for live data streaming.
* **Deep Learning Adaptation:** Implement 1D Convolutional Neural Networks (1D-CNNs) and Transformers to extract finer spectral signature patterns from dense wavelength data.
* **Edge Computing Deployment:** Optimize and compress models using TensorFlow Lite to run calculations directly on microcontrollers or edge gateways without internet reliance.
* **Dynamic Fertilizer Recommendation System:** Add an prescriptive analytics layer that suggests exact real-time N-P-K fertilizer dosages based on the predicted soil nutrient deficit.
* **Geospatial Mapping:** Integrate GIS mapping and GPS tracking to visualize soil health degradation heatmaps over large geographic farmlands.

---

## 🛠️ Getting Started

### Prerequisites
Ensure you have the following packages installed:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Execution
Run the main pipeline execution file:
```bash

---
Thank you !!
python main.py
```
