import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import joblib
import os

heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
df = X.copy()
df['target'] = y

print("Initial data exploration:")
print(df.head())
print(df.describe())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Target distribution:\n", df['target'].value_counts())

df.replace('?', np.nan, inplace=True)


numeric_cols = ['age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate', 'oldpeak']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns


num_cols_to_impute = [col for col in numerical_cols if col != 'target']
if len(num_cols_to_impute) > 0:
    df[num_cols_to_impute] = num_imputer.fit_transform(df[num_cols_to_impute])

if len(categorical_cols) > 0:
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])


df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


X = df.drop('target', axis=1)
y = df['target']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)




cleaned_df = X_scaled.copy()
cleaned_df['target'] = y.values  

print("Final target ")
print(cleaned_df['target'].value_counts())


os.makedirs("C:/studies/sprints/Heart_Disease_Project/data", exist_ok=True)


cleaned_df.to_csv("C:/studies/sprints/Heart_Disease_Project/data/heart_disease_cleaned.csv", index=False)


os.makedirs('./models', exist_ok=True)  
joblib.dump(scaler, 'C:/studies/sprints/Heart_Disease_Project/models/scaler.pkl')

print("Scaler saved to 'models/scaler.pkl' in current directory!")


plt.figure(figsize=(12, 8))
cleaned_df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = cleaned_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
cleaned_df.boxplot(figsize=(12, 8))
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()