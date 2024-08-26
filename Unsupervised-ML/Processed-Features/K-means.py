import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
import os 

# 1 - main pipeline 

def load_and_clean_data(file_path):
    """Load dataset, drop duplicates, and separate binary and continuous features."""
    data = pd.read_csv(file_path, sep='\t')
    data_cleaned = data.drop_duplicates()
    
    binary_columns = data_cleaned.select_dtypes(include=['int', 'bool']).columns.tolist()
    continuous_columns = data_cleaned.select_dtypes(include='float').columns.tolist()
    
    return data_cleaned, binary_columns, continuous_columns

def impute_and_transform(data, continuous_columns):
    """Impute missing values and apply rank-based inverse transformation to continuous features."""
    imputer = SimpleImputer(strategy='mean')
    data[continuous_columns] = imputer.fit_transform(data[continuous_columns])
    
    n_samples = data.shape[0]
    rank_transformer = QuantileTransformer(output_distribution='normal', n_quantiles=min(n_samples, 1000), random_state=42)
    transformed_continuous = pd.DataFrame(rank_transformer.fit_transform(data[continuous_columns]), columns=continuous_columns)
    
    return transformed_continuous

def remove_outliers_zscore(data, continuous_columns, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs((data[continuous_columns] - data[continuous_columns].mean()) / data[continuous_columns].std())
    data_out = data[(z_scores < threshold).all(axis=1)]
    return data_out

def perform_pca(features, n_components=3):
    """Perform PCA and reduce features to the specified number of components."""
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    return features_pca

def determine_optimal_k(features_pca, k_values):
    """Determine the optimal number of clusters using silhouette score and Davies-Bouldin Index."""
    silhouette_scores = []
    dbi_scores = []
    wcss = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(features_pca)
        
        silhouette_avg = silhouette_score(features_pca, clusters)
        dbi_score = davies_bouldin_score(features_pca, clusters)
        silhouette_scores.append(silhouette_avg)
        dbi_scores.append(dbi_score)
        wcss.append(kmeans.inertia_)
    
    results_df = pd.DataFrame({'k': k_values, 'Silhouette Score': silhouette_scores, 'DBI': dbi_scores})
    return results_df, wcss

def plot_scores(k_values, silhouette_scores, dbi_scores, wcss):
    """Plot silhouette scores, Davies-Bouldin Index, and elbow method results."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, dbi_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('DBI for Optimal k')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.plot(k_values, wcss, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def perform_clustering(features_pca, optimal_k):
    """Perform K-Means clustering and return cluster assignments."""
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    clusters_optimal = kmeans_optimal.fit_predict(features_pca)
    return clusters_optimal

def plot_clusters_2d(features_pca, clusters):
    """Plot 2D scatter plot of clusters."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=clusters, palette='tab10', s=100)
    plt.title('K-Means Clusters on Processed Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.show()

def main():
    file_path = '/Users/cheaulee/Desktop/hfproject/DATA/processed_feature_table.tsv'  
    data_cleaned, binary_columns, continuous_columns = load_and_clean_data(file_path)
    
    data_cleaned = remove_outliers_zscore(data_cleaned, continuous_columns)
    transformed_continuous = impute_and_transform(data_cleaned, continuous_columns)
    
    features_transformed = transformed_continuous.join(data_cleaned[binary_columns].reset_index(drop=True))
    features_transformed = features_transformed.dropna()
    
    features_pca = perform_pca(features_transformed)

    k_values = range(2, 11)
    results_df, wcss = determine_optimal_k(features_pca, k_values)

    plot_scores(k_values, results_df['Silhouette Score'], results_df['DBI'], wcss)

    optimal_k = k_values[np.argmax(results_df['Silhouette Score'])]
    clusters_optimal = perform_clustering(features_pca, optimal_k)

    results = pd.DataFrame({'Gene_Name': data_cleaned['Gene_Name'].reset_index(drop=True), 'Cluster': clusters_optimal + 1})  # Adjust cluster numbering to start from 1

    plot_clusters_2d(features_pca, results['Cluster'])

    print("K-Means Clustering Results")
    print(results_df)

    print("First few rows of the clustered results:")
    print(results.head())

    output_clustered_file = f'/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pfgeneclusters.csv'
    results.to_csv(output_clustered_file, index=False)

    print(f"Clustered genes with K-Means Clustering (k={optimal_k}) have been saved to {output_clustered_file}")

if __name__ == "__main__":
    main()

# 2 - display gene counts per cluster

# Read the CSV file
csv_file = '/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pfgeneclusters.csv'
df = pd.read_csv(csv_file)

# Group the data by clusters and count the number of genes in each cluster
cluster_counts = df.groupby('Cluster')['Gene_Name'].count()

# Display the number of genes in each cluster
print(cluster_counts)

# 3 - save gene lists per cluster

# Group the data by clusters
grouped = df.groupby('Cluster')

# Define the destination directory
destination_dir = '/Users/cheaulee/Desktop/hfproject/PF/Kmeans/'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Save each cluster's gene list to a separate text file
for cluster, data in grouped:
    gene_list = data['Gene_Name'].tolist()
    file_path = os.path.join(destination_dir, f'pf_cluster_{cluster}.txt')
    with open(file_path, 'w') as file:
        for gene in gene_list:
            file.write(f'{gene}\n')

print("Files saved successfully.")
