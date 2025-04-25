# Anomaly Detection Using Machine Learning

## Project Overview

This project addresses anomaly detection in manufacturing processes, specifically injection molding, utilizing machine learning techniques. The goal is to detect anomalies early, minimizing defects and associated costs. The analysis includes data preprocessing, exploratory visualizations, statistical testing, and developing classification models such as Random Forest, ANN, Logistic Regression, Extra Trees, and Naive Bayes.

## Objectives

- **Detect Anomalies Early:** Use sensor data to detect production cycle anomalies.
- **Improve Manufacturing Quality:** Reduce defect rates through accurate anomaly prediction.
- **Create Actionable Insights:** Identify critical factors influencing anomalies for process optimization.
- **Deploy an Interactive Dashboard:** Provide a user-friendly interface for real-time monitoring and intervention.

## Dataset

The dataset contains 1,000 production cycle observations from injection molding processes, featuring:
- Melt and mold temperatures
- Cycle times and plasticizing times
- Forces and torque metrics
- Injection and back pressures
- Quality labels (categorized from 1 to 4)

## Methodology

1. **Data Preprocessing:**
   - Exploratory analysis and visualization
   - Pairwise correlation matrix
   - Handling outliers using the Interquartile Range (IQR)
   - Data scaling (standardization)

2. **Feature Importance Analysis:**
   - ANOVA tests to identify significant predictors of anomalies

3. **Model Development:**
   - Models: ANN, Random Forest, Logistic Regression, Extra Trees, Naive Bayes
   - Training with 10-fold cross-validation
   - Hyperparameter tuning using GridSearchCV

4. **Model Evaluation:**
   - Metrics: accuracy, precision, recall, F1-score
   - Confusion matrices for visual evaluation

## Results

The Random Forest and Extra Trees classifiers demonstrated top performance (~94% accuracy). The deployed Streamlit dashboard provides interactive real-time anomaly detection, enabling proactive decision-making and process optimization.

## Repository Structure



# Product Quality Prediction Dashboard

This Streamlit app allows real-time predictions of manufacturing product quality using an uploaded trained model (.pkl) and a dataset (.csv).

## Features

- Upload model and dataset directly from the interface
- Real-time prediction and classification
- Confusion matrix and evaluation metrics
- SHAP-based model explainability

## To Deploy on Streamlit Cloud

1. Upload this repository to your GitHub account
2. Go to https://streamlit.io/cloud
3. Click "New App" and select this repo and `app_upload_model.py`
4. Deploy your interactive dashboard live!
