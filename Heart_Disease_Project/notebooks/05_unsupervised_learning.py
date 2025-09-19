
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

def perform_unsupervised_learning():
    print("Performing unsupervised learning...")
    
   
    pca_df = pd.read_csv("C:\studies\sprints\Heart_Disease_Project\data\heart_disease_pca.csv")
    X_pca = pca_df.drop('target', axis=1)
    y = pca_df['target']

    
    print("Performing K-Means clustering")
    inertia = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('C:/studies/sprints/Heart_Disease_Project/results/kmeans_elbow_plot.png')
    plt.close()

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca)

    print("Performing Hierarchical clustering")
    linked = linkage(X_pca, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig('C:/studies/sprints/Heart_Disease_Project/results/hierarchical_dendrogram.png')
    plt.close()

    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X_pca)

    clustering_results = {
        'K-Means': {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X_pca, kmeans_labels),
            'ari': adjusted_rand_score(y, kmeans_labels)
        },
        'Hierarchical': {
            'labels': hierarchical_labels,
            'silhouette': silhouette_score(X_pca, hierarchical_labels),
            'ari': adjusted_rand_score(y, hierarchical_labels)
        }
    }

    print("\nClustering Results:")
    for method, metrics in clustering_results.items():
        print(f"{method}:")
        print(f"  Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"  Adjusted Rand Index: {metrics['ari']:.4f}")

    clustering_df = pd.DataFrame({
        'kmeans_labels': kmeans_labels,
        'hierarchical_labels': hierarchical_labels,
        'true_labels': y
    })
    clustering_df.to_csv('C:/studies/sprints/Heart_Disease_Project/results/clustering_results.csv', index=False)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca.iloc[:, 0], X_pca.iloc[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-Means Clustering Results')

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca.iloc[:, 0], X_pca.iloc[:, 1], c=hierarchical_labels, cmap='plasma', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Hierarchical Clustering Results')

    plt.tight_layout()
    plt.savefig('C:/studies/sprints/Heart_Disease_Project/results/clustering_visualization.png')
    plt.close()

    return clustering_results

if __name__ == "__main__":
    perform_unsupervised_learning()