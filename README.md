# Employee_salary_prediction_
A Flask-based web app that predicts whether an employee earns more than $50K per year using machine learning on census data.

# 👩‍💼 Employee Income Prediction Web App

This is a machine learning web application built using **Flask** that predicts whether an individual earns more than $50K per year based on demographic features from the **Adult Census Income dataset**.

## 🔍 Project Overview

The application takes user input through a web form — including age, work class, education, marital status, occupation, race, gender, hours worked per week, and native country — and uses a trained machine learning model to classify income as:

- **`>50K` (more than 50,000 USD annually)** or
- **`<=50K` (50,000 USD or less annually)**

## 🧠 Machine Learning Details

- Dataset: Adult Census Income Dataset (`adult.csv`)
- Preprocessing:
  - Null value handling
  - Label encoding and one-hot encoding
  - Feature selection
- Models tried:
  - Logistic Regression
  - K-Nearest Neighbors
  - Random Forest Classifier ✅ (Final model)
- Evaluation: Accuracy score, confusion matrix, and classification report

## 💻 Tech Stack

- Python
- Flask (Backend)
- HTML & CSS (Frontend)
- Scikit-learn (ML model)
- Pandas, NumPy (Data processing)
- Matplotlib/Seaborn (Visualization)

