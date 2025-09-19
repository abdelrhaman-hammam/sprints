
# Heart Disease Prediction Model Card

## Model Information
- **Model Type**: RandomForestClassifier
- **Version**: 1.0
- **Training Date**: 2025-09-19 07:47:53
- **Best Accuracy**: 0.8689

## Hyperparameters
{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 30}

## Features
Number of features: 12
Feature names: ['oldpeak', 'thal', 'sex', 'trestbps', 'ca', 'cp', 'chol', 'slope', 'exang', 'thalach', 'restecg', 'age']

## Training Data
- Samples: 303
- Classes: [0, 1]

## Performance Metrics
- Best CV Score: 0.8689
- Test Accuracy: 0.8689

## Deployment Instructions
1. Load model: `joblib.load('model_package.pkl')`
2. Preprocess input data same as training
3. Use model.predict() for predictions

## Model Limitations
- Trained on specific heart disease dataset
- Accuracy may vary with different populations
- Requires same feature preprocessing as training data
