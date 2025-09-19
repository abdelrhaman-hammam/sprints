import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def perform_hyperparameter_tuning():
    print("Performing hyperparameter tuning")
    
    
    df = pd.read_csv("C:/studies/sprints/Heart_Disease_Project/data/heart_disease_selected.csv")
    X = df.drop('target', axis=1)
    y = df['target']

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = RandomizedSearchCV(
        rf, param_grid, n_iter=50, cv=5, scoring='accuracy', 
        n_jobs=-1, random_state=42
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)

    
    y_pred_best = best_rf.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"Best model test accuracy: {best_accuracy:.4f}")

    
    print("\n" + "="*50)
    print("MODEL EXPORT & DEPLOYMENT")
    print("="*50)
    
    
    os.makedirs("C:/studies/sprints/Heart_Disease_Project/models", exist_ok=True)
    os.makedirs("C:/studies/sprints/Heart_Disease_Project/deployment", exist_ok=True)

    
    model_path = "C:/studies/sprints/Heart_Disease_Project/models/final_model.pkl"
    joblib.dump(best_rf, model_path)
    print(f"✅ Model saved to: {model_path}")

    
    deployment_package = {
        'model': best_rf,
        'model_type': 'RandomForestClassifier',
        'model_version': '1.0',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_names': list(X.columns),
        'best_parameters': grid_search.best_params_,
        'best_accuracy': best_accuracy,
        'model_metadata': {
            'n_features': X.shape[1],
            'n_samples': len(X),
            'classes': best_rf.classes_.tolist() if hasattr(best_rf, 'classes_') else None,
            'n_estimators': best_rf.n_estimators if hasattr(best_rf, 'n_estimators') else None
        }
    }

    deployment_path = "C:/studies/sprints/Heart_Disease_Project/deployment/model_package.pkl"
    joblib.dump(deployment_package, deployment_path)
    print(f" Deployment package saved to: {deployment_path}")

    
    save_model_card(deployment_package, best_accuracy, X_test, y_test, y_pred_best)
    
    
    tuning_results = pd.DataFrame(grid_search.cv_results_)
    tuning_path = "C:/studies/sprints/Heart_Disease_Project/results/hyperparameter_tuning_results.csv"
    tuning_results.to_csv(tuning_path, index=False)
    print(f" Tuning results saved to: {tuning_path}")

    print("\n DEPLOYMENT READY!")
    print("Files created for deployment:")
    print(f"   - {model_path} (Main model)")
    print(f"   - {deployment_path} (Complete deployment package)")
    print(f"   - {tuning_path} (Tuning results)")
    
    return best_rf, best_accuracy

def save_model_card(deployment_package, accuracy, X_test, y_test, y_pred):
    """Save model documentation and performance metrics"""
    model_card = f"""
# Heart Disease Prediction Model Card

## Model Information
- **Model Type**: {deployment_package['model_type']}
- **Version**: {deployment_package['model_version']}
- **Training Date**: {deployment_package['training_date']}
- **Best Accuracy**: {deployment_package['best_accuracy']:.4f}

## Hyperparameters
{deployment_package['best_parameters']}

## Features
Number of features: {deployment_package['model_metadata']['n_features']}
Feature names: {deployment_package['feature_names']}

## Training Data
- Samples: {deployment_package['model_metadata']['n_samples']}
- Classes: {deployment_package['model_metadata']['classes']}

## Performance Metrics
- Best CV Score: {deployment_package['best_accuracy']:.4f}
- Test Accuracy: {accuracy:.4f}

## Deployment Instructions
1. Load model: `joblib.load('model_package.pkl')`
2. Preprocess input data same as training
3. Use model.predict() for predictions

## Model Limitations
- Trained on specific heart disease dataset
- Accuracy may vary with different populations
- Requires same feature preprocessing as training data
"""

    model_card_path = "C:/studies/sprints/Heart_Disease_Project/deployment/MODEL_CARD.md"
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    print(f"✅ Model card saved to: {model_card_path}")

if __name__ == "__main__":
    perform_hyperparameter_tuning()