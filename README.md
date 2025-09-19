Developed as part of the Sprints Program, this project is a machine learning pipeline for heart disease prediction. Utilizing clinical data from the UCI Machine Learning Repository (Heart Disease Dataset, featuring 13 clinical features such as age, blood pressure, and cholesterol levels; target: presence of heart disease), the application predicts heart disease likelihood with high accuracy, offering a valuable decision-support tool for healthcare professionals. The system encompasses data processing and cleaning, feature selection and dimensionality reduction, training of multiple machine learning models with hyperparameter optimization, and deployment of a user-friendly Streamlit web application for real-time risk predictions. This end-to-end solution was built following industry best practices and production-level standards during an intensive training program.

1. Data Preparation
Run 01_data_preprocessing.py - cleans and prepares data
Handles missing values and formatting
Creates scaled features for ML

2. Feature Engineering
Run 02_pca_analysis.py - reduces dimensionality
Run 03_feature_selection.py - selects important features

3. Model Training
Run 04_supervised_learning.py - trains 4 ML models
Run 06_hyperparameter_tuning.py - optimizes best model

4. Use Application
Launch with streamlit run app.py
Fill in patient information in sidebar
Click "Predict Heart Disease Risk"
View results and recommendations
