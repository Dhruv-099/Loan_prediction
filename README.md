# Loan Status Prediction

This project aims to develop a machine learning model to predict the loan status (approved or rejected) based on various features such as applicant income, credit history, loan amount, and other relevant factors. Additionally, a FastAPI application has been developed to serve the trained model for making predictions.

## Dataset

The dataset used in this project is the "Loan Status Prediction" dataset from Kaggle. It contains information about loan applications, including the applicant's personal details, employment details, and loan-related information.

## Code Structure

The project consists of the following files:

1. **app.py**: This file contains a FastAPI application that exposes an endpoint `/predict/` for making predictions using the trained model.
2. **loan - Kaggle.ipynb**: This Jupyter Notebook contains the code for data exploration, preprocessing, feature engineering, model training, and evaluation.

## Requirements

To run this project, you'll need the following dependencies:

- Python (version 3.6 or higher)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- FastAPI
- Joblib (for model serialization)
## Model Performance
The notebook (loan - Kaggle.ipynb) includes code for training and evaluating multiple machine learning models, such as Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), AdaBoost, and Random Forest Classifier. The performance of each model is reported using accuracy as the evaluation metric.
