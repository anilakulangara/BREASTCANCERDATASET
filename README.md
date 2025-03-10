# BREAST CANCER DATASET

📌 Project Overview

This project aims to classify breast cancer tumors as malignant or benign using machine learning algorithms. The dataset used is the Breast Cancer Wisconsin Dataset from sklearn.datasets.

🎯 Objectives

Load and preprocess the dataset to ensure data quality.

Implement multiple classification models to predict cancer malignancy.

Compare the performance of these models to determine the best one.

Dataset

The dataset contains 30 numeric features extracted from digitized images of breast masses. The target variable is binary (0 = Malignant, 1 = Benign).

📌 Source: sklearn.datasets.load_breast_cancer()

Key Steps

Data Loading & Preprocessing

✅ Load the dataset using sklearn.datasets.load_breast_cancer().
✅ Convert it into a Pandas DataFrame.
✅ Check for missing values (none found).
✅ Apply feature scaling using StandardScaler() to normalize data.

 Model Implementation

We implement and evaluate five classification algorithms:

-Logistic Regression

-Decision Tree Classifier

-Random Forest Classifier

-Support Vector Machine (SVM)

-k-Nearest Neighbors (k-NN)

 📈 Results & Best Model
 K-Nearest Neighbors (k-NN) is the best model as it has the lowest MSE and MAE and the highest R² Score (0.8544).
