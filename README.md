Temperature Prediction Model
A machine learning project that predicts average temperatures using ensemble and individual regression models.
Overview
This repository contains a comprehensive analysis and prediction system for average temperature data. The project implements multiple regression algorithms and combines them into an ensemble model to achieve optimal prediction accuracy.
Features

Data exploration and preprocessing
Feature engineering with time-based variables
Multiple prediction models:

Linear Regression
Random Forest Regression
Gradient Boosting Regression
Support Vector Regression
Ensemble model (Voting Regressor)


Detailed model evaluation and comparison
Feature importance analysis
Visualization of results

Requirements

Python 3.x
pandas
numpy
matplotlib
scikit-learn
seaborn

Dataset
The project uses a temperature dataset that should be saved as temperature.csv in your Downloads folder. The main target variable is Avg_Temperature_degC.
Usage

Ensure the temperature dataset is in the correct location
Run the script:

python temperature_prediction.py
Output
The script generates several visualization files:

rf_feature_importances.png: Random Forest feature importance
gb_feature_importances.png: Gradient Boosting feature importance
model_comparison.png: Performance comparison of all models
actual_vs_predicted.png: Scatter plot of actual vs predicted temperatures
prediction_errors.png: Distribution of prediction errors
model_predictions_comparison.png: Comparison of all model predictions
correlation_matrix.png: Correlation matrix of climate variables

Results
Models are evaluated using RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and R² metrics. The script automatically identifies the best performing model based on RMSE.
Future Work

Temporal analysis to track temperature trends
Hyperparameter tuning for improved performance
Integration of additional climate variables
Deploying the model as a web service

Contribution
Feel free to fork this repository and contribute to the project. Pull requests are welcome!
