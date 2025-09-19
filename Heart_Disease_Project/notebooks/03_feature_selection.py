import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib
def perform_feature_selection():
    print("Performing feature selection...")
    
   
    df = pd.read_csv("C:/studies/sprints/Heart_Disease_Project/data/heart_disease_cleaned.csv")
    X = df.drop('target', axis=1)
    y = df['target']

    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.close()

    
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X, y)

    xgb_importances = xgb.feature_importances_
    xgb_indices = np.argsort(xgb_importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances (XGBoost)")
    plt.bar(range(X.shape[1]), xgb_importances[xgb_indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in xgb_indices], rotation=90)
    plt.tight_layout()
    plt.close()

    
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    selector = selector.fit(X, y)

    rfe_selected = X.columns[selector.support_]
    print("RFE selected features:", list(rfe_selected))

    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    chi2_selector = SelectKBest(chi2, k=10)
    chi2_selector.fit(X_scaled, y)

    chi2_selected = X.columns[chi2_selector.get_support()]
    print("Chi-square selected features:", list(chi2_selected))

    
    selected_features = list(set().union(
        rfe_selected, 
        chi2_selected, 
        [feature_names[i] for i in indices[:10]],
        [feature_names[i] for i in xgb_indices[:10]]
    ))
    
    print("Final selected features:", selected_features)

    
    X_selected = X[selected_features]
    selected_df = X_selected.copy()
    selected_df['target'] = y
    
    # Save selected features dataset
    selected_df.to_csv('C:\studies\sprints\Heart_Disease_Project\data\heart_disease_selected.csv', index=False)
    print("Selected features dataset saved to '../data/heart_disease_selected.csv'")
    
    # Save feature importance results
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_rf': importances,
        'importance_xgb': xgb_importances,
        'rfe_selected': [1 if feat in rfe_selected else 0 for feat in feature_names],
        'chi2_selected': [1 if feat in chi2_selected else 0 for feat in feature_names]
    })
    
    feature_importance_df.to_csv("C:\studies\sprints\Heart_Disease_Project\data\heart_disease_feature_results.csv", index=False)
    print("Feature importance results saved to '../results/feature_importance_results.csv'")

    return selected_df

if __name__ == "__main__":
    perform_feature_selection()