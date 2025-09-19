import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv("C:/studies/sprints/Heart_Disease_Project/data/heart_disease_cleaned.csv")
X = df.drop('target', axis=1)
y = df['target']

print("Original dataset ", X.shape)
print("Number of features:", X.shape[1])

pca = PCA()
X_pca = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance by each component:")
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {ev:.4f} (Cumulative: {cv:.4f})")

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f'\nNumber of components explaining 95% variance: {n_components}')
print(f'Variance explained by {n_components} components: {cumulative_variance[n_components-1]:.4f}')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='blue')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Explained Variance')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='red')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.8)
plt.axvline(x=n_components, color='gray', linestyle='--', alpha=0.8)

plt.tight_layout()
plt.show()

pca_optimal = PCA(n_components=n_components)
X_pca_reduced = pca_optimal.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y, 
                     cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter, label='Heart Disease (0=No, 1=Yes)')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.title('PCA Projection of Heart Disease Dataset')
plt.grid(True, alpha=0.3)

variance_pc1 = explained_variance[0] * 100
variance_pc2 = explained_variance[1] * 100
plt.figtext(0.15, 0.82, f'PC1 explains {variance_pc1:.1f}% of variance', 
           fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.figtext(0.15, 0.78, f'PC2 explains {variance_pc2:.1f}% of variance', 
           fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()

pca_columns = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(X_pca_reduced, columns=pca_columns)
pca_df['target'] = y.values


pca_df.to_csv('C:/studies/sprints/Heart_Disease_Project/data/heart_disease_pca.csv', index=False)
print(f"PCA-transformed dataset saved with {n_components} components")

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features: {n_components}")
print(f"Variance retained: {cumulative_variance[n_components-1]:.4f} ({cumulative_variance[n_components-1]*100:.2f}%)")


loadings = pd.DataFrame(
    pca_optimal.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=X.columns
)

print("\nTop features contributing to PC1 and PC2:")
top_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(5)
top_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(5)

print("Top features in PC1:")
for feature, loading in top_pc1.items():
    print(f"  {feature}: {loading:.4f}")

print("\nTop features in PC2:")
for feature, loading in top_pc2.items():
    print(f"  {feature}: {loading:.4f}")

