import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

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

def perform_hierarchical_clustering(features_pca, n_clusters):
    """Perform hierarchical clustering and return cluster assignments."""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical.fit_predict(features_pca)
    return clusters

def evaluate_hierarchical_clustering(features_pca, n_clusters_range):
    """Evaluate hierarchical clustering using silhouette and DBI scores for different numbers of clusters."""
    silhouette_scores = []
    dbi_scores = []

    for n_clusters in n_clusters_range:
        clusters = perform_hierarchical_clustering(features_pca, n_clusters)
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(features_pca, clusters)
            dbi_score = davies_bouldin_score(features_pca, clusters)
        else:
            silhouette_avg = -1  # Assign a poor score if clustering fails
            dbi_score = np.inf
        
        silhouette_scores.append(silhouette_avg)
        dbi_scores.append(dbi_score)
    
    results_df = pd.DataFrame({'Number of Clusters': n_clusters_range, 'Silhouette Score': silhouette_scores, 'DBI': dbi_scores})
    return results_df

def plot_scores(n_clusters_range, silhouette_scores, dbi_scores):
    """Plot silhouette scores and Davies-Bouldin Index for different numbers of clusters."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(n_clusters_range, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters')

    plt.subplot(1, 2, 2)
    plt.plot(n_clusters_range, dbi_scores, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('DBI for Different Numbers of Clusters')

    plt.tight_layout()
    plt.show()

def plot_dendrogram(data):
    """Plot dendrogram for hierarchical clustering."""
    linked = linkage(data, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

def plot_clusters_2d(features_pca, clusters):
    """Plot 2D scatter plot of clusters."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=clusters, palette='tab10', s=100)
    plt.title('Hierarchical Clusters on Processed Features')
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

    # Plot dendrogram
    plot_dendrogram(features_pca)

    # Define a range of cluster numbers to evaluate
    n_clusters_range = range(2, 11)

    # Evaluate hierarchical clustering
    results_df = evaluate_hierarchical_clustering(features_pca, n_clusters_range)
    plot_scores(results_df['Number of Clusters'], results_df['Silhouette Score'], results_df['DBI'])

    # Print evaluation results in a table format
    print("Evaluation Results for Hierarchical Clustering:")
    print(results_df)

    # Normalize the scores and calculate combined score
    results_df['Silhouette Score Norm'] = (results_df['Silhouette Score'] - results_df['Silhouette Score'].min()) / (results_df['Silhouette Score'].max() - results_df['Silhouette Score'].min())
    results_df['DBI Norm'] = (results_df['DBI'].max() - results_df['DBI']) / (results_df['DBI'].max() - results_df['DBI'].min())
    results_df['Combined Score'] = (results_df['Silhouette Score Norm'] + results_df['DBI Norm']) / 2

    optimal_n_clusters_combined = results_df.loc[results_df['Combined Score'].idxmax(), 'Number of Clusters']

    print(f"Optimal number of clusters based on combined Silhouette Score and DBI: {optimal_n_clusters_combined}")

    clusters_optimal_combined = perform_hierarchical_clustering(features_pca, optimal_n_clusters_combined)

    results_combined = pd.DataFrame({'Gene_Name': data_cleaned['Gene_Name'].reset_index(drop=True), 'Cluster': clusters_optimal_combined + 1})  # Adjust cluster numbering to start from 1

    print("First few rows of the clustered results based on combined Silhouette Score and DBI:")
    print(results_combined.head())

    plot_clusters_2d(features_pca, results_combined['Cluster'])

    # Save the clustered results to a CSV file
    output_clustered_file_combined = f'/Users/cheaulee/Desktop/hfproject/PF/Hierarchical/pfgeneclusters_hierarchical.csv'
    results_combined.to_csv(output_clustered_file_combined, index=False)

    print(f"Clustered genes with hierarchical clustering (combined Silhouette Score and DBI) have been saved to {output_clustered_file_combined}")

if __name__ == "__main__":
    main()
