import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
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

def determine_optimal_dbscan_params(features_pca):
    """Determine the optimal epsilon for DBSCAN using the k-distance graph."""
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors_fit = neighbors.fit(features_pca)
    distances, indices = neighbors_fit.kneighbors(features_pca)
    distances = np.sort(distances[:, 9], axis=0)
    return distances

def plot_k_distance(distances):
    """Plot k-distance graph to determine the optimal epsilon."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('Epsilon')
    plt.title('K-distance Graph for DBSCAN')
    plt.show()

def perform_dbscan(features_pca, eps, min_samples=5):
    """Perform DBSCAN clustering and return cluster assignments."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features_pca)
    return clusters

def evaluate_dbscan(features_pca, eps_values, min_samples=5):
    """Evaluate DBSCAN clustering using silhouette and DBI scores for different epsilon values."""
    silhouette_scores = []
    dbi_scores = []

    for eps in eps_values:
        clusters = perform_dbscan(features_pca, eps, min_samples)
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(features_pca, clusters)
            dbi_score = davies_bouldin_score(features_pca, clusters)
        else:
            silhouette_avg = -1  # Assign a poor score if clustering fails
            dbi_score = np.inf
        
        silhouette_scores.append(silhouette_avg)
        dbi_scores.append(dbi_score)
    
    results_df = pd.DataFrame({'Epsilon': eps_values, 'Silhouette Score': silhouette_scores, 'DBI': dbi_scores})
    return results_df

def plot_scores(eps_values, silhouette_scores, dbi_scores):
    """Plot silhouette scores and Davies-Bouldin Index for different epsilon values."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(eps_values, silhouette_scores, 'bx-')
    plt.xlabel('Epsilon')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Epsilon Values')

    plt.subplot(1, 2, 2)
    plt.plot(eps_values, dbi_scores, 'bx-')
    plt.xlabel('Epsilon')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('DBI for Different Epsilon Values')

    plt.tight_layout()
    plt.show()

def plot_clusters_2d(features_pca, clusters):
    """Plot 2D scatter plot of clusters."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=clusters, palette='tab10', s=100)
    plt.title('DBSCAN Clusters on Processed Features')
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

    distances = determine_optimal_dbscan_params(features_pca)
    plot_k_distance(distances)

    # Define a range of epsilon values to evaluate
    eps_values = np.arange(0.5, 5.0, 0.5)
    min_samples = 5  # Default value, can be adjusted

    results_df = evaluate_dbscan(features_pca, eps_values, min_samples)
    plot_scores(results_df['Epsilon'], results_df['Silhouette Score'], results_df['DBI'])

    # Print evaluation results in a table format
    print("Evaluation Results for DBSCAN:")
    print(results_df)

    # Normalize the scores and calculate combined score
    results_df['Silhouette Score Norm'] = (results_df['Silhouette Score'] - results_df['Silhouette Score'].min()) / (results_df['Silhouette Score'].max() - results_df['Silhouette Score'].min())
    results_df['DBI Norm'] = (results_df['DBI'].max() - results_df['DBI']) / (results_df['DBI'].max() - results_df['DBI'].min())
    results_df['Combined Score'] = (results_df['Silhouette Score Norm'] + results_df['DBI Norm']) / 2

    optimal_eps_combined = results_df.loc[results_df['Combined Score'].idxmax(), 'Epsilon']

    print(f"Optimal epsilon based on combined Silhouette Score and DBI: {optimal_eps_combined}")

    clusters_optimal_combined = perform_dbscan(features_pca, optimal_eps_combined, min_samples)

    results_combined = pd.DataFrame({'Gene_Name': data_cleaned['Gene_Name'].reset_index(drop=True), 'Cluster': clusters_optimal_combined + 1})  # Adjust cluster numbering to start from 1

    print("First few rows of the clustered results based on combined Silhouette Score and DBI:")
    print(results_combined.head())

    plot_clusters_2d(features_pca, results_combined['Cluster'])

    # Save the clustered results to a CSV file
    output_clustered_file_combined = f'/Users/cheaulee/Desktop/hfproject/PF/DBSCAN/pfgeneclusters_dbscan.csv'
    results_combined.to_csv(output_clustered_file_combined, index=False)

    print(f"Clustered genes with DBSCAN (combined Silhouette Score and DBI) have been saved to {output_clustered_file_combined}")

if __name__ == "__main__":
    main()
