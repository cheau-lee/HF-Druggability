import pandas as pd


# Common genes between Open Targets and DrugnomeAI

def load_genes(file_path):
    """Load genes from a CSV file."""
    genes_df = pd.read_csv(file_path, header=None)
    genes = set(genes_df[0].tolist())
    return genes

def load_known_drugs(file_path):
    """Load known drugs dataset and extract unique gene symbols."""
    known_drugs_df = pd.read_csv(file_path, sep='\t')
    known_genes = set(known_drugs_df['symbol'].tolist())
    return known_genes, known_drugs_df

def find_overlapping_genes(genes, known_genes):
    """Find overlapping genes between the two sets."""
    overlapping_genes = genes.intersection(known_genes)
    return overlapping_genes

def main():
    # File paths
    genes_file_path = '/Users/cheaulee/Desktop/hfproject/DATA/drugnomeai_filtered_genes.csv'
    known_drugs_file_path = '/Users/cheaulee/Desktop/hfproject/DATA/OpenTargets/EFO_0003144-known-drugs.tsv'
    
    # Load genes
    genes = load_genes(genes_file_path)
    print(f"Loaded {len(genes)} genes from {genes_file_path}")
    
    # Load known drugs data
    known_genes, known_drugs_df = load_known_drugs(known_drugs_file_path)
    print(f"Loaded {len(known_genes)} known genes from {known_drugs_file_path}")
    
    # Find overlapping genes
    overlapping_genes = find_overlapping_genes(genes, known_genes)
    print(f"Found {len(overlapping_genes)} overlapping genes:")
    
    # Print overlapping genes
    for gene in overlapping_genes:
        print(gene)
    
    # Filter known drugs dataframe for overlapping genes
    overlapping_drugs_df = known_drugs_df[known_drugs_df['symbol'].isin(overlapping_genes)]
    
    # Save overlapping genes and corresponding rows to a file
    output_file_path = '/Users/cheaulee/Desktop/hfproject/DF/overlapping_OT_DrugnomeAI.csv'
    overlapping_drugs_df.to_csv(output_file_path, index=False)
    print(f"Rows for overlapping genes have been saved to {output_file_path}")

if __name__ == "__main__":
    main()





# Common genes between Open Targets and MAGMA

def load_genes(file_path):
    """Load genes from a CSV file."""
    genes_df = pd.read_csv(file_path, header=None)
    genes = set(genes_df[0].tolist())
    return genes

def load_known_drugs(file_path):
    """Load known drugs dataset and extract unique gene symbols."""
    known_drugs_df = pd.read_csv(file_path, sep='\t')
    known_genes = set(known_drugs_df['symbol'].tolist())
    return known_genes, known_drugs_df

def find_overlapping_genes(genes, known_genes):
    """Find overlapping genes between the two sets."""
    overlapping_genes = genes.intersection(known_genes)
    return overlapping_genes

def main():
    # File paths
    genes_file_path = '/Users/cheaulee/Desktop/hfproject/DATA/MAGMA_genes.txt'
    known_drugs_file_path = '/Users/cheaulee/Desktop/hfproject/DATA/OpenTargets/EFO_0003144-known-drugs.tsv'
    
    # Load genes
    genes = load_genes(genes_file_path)
    print(f"Loaded {len(genes)} genes from {genes_file_path}")
    
    # Load known drugs data
    known_genes, known_drugs_df = load_known_drugs(known_drugs_file_path)
    print(f"Loaded {len(known_genes)} known genes from {known_drugs_file_path}")
    
    # Find overlapping genes
    overlapping_genes = find_overlapping_genes(genes, known_genes)
    print(f"Found {len(overlapping_genes)} overlapping genes:")
    
    # Print overlapping genes
    for gene in overlapping_genes:
        print(gene)
    
    # Filter known drugs dataframe for overlapping genes
    overlapping_drugs_df = known_drugs_df[known_drugs_df['symbol'].isin(overlapping_genes)]
    
    # Save overlapping genes and corresponding rows to a file
    output_file_path = '/Users/cheaulee/Desktop/hfproject/DATA/OpenTargets/overlapping_OT_MAGMA.csv'
    overlapping_drugs_df.to_csv(output_file_path, index=False)
    print(f"Rows for overlapping genes have been saved to {output_file_path}")

if __name__ == "__main__":
    main()





# Cluster pattern for processed features 

def load_known_drugs(file_path):
    """Load known drugs dataset and extract unique gene symbols."""
    known_drugs_df = pd.read_csv(file_path, sep='\t')
    known_genes = set(known_drugs_df['symbol'].tolist())
    return known_genes, known_drugs_df

def main():
    # Paths to the input files
    known_drugs_file_path = "/Users/cheaulee/Desktop/hfproject/DATA/OpenTargets/EFO_0003144-known-drugs.tsv"
    cluster_files = [
        "/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pf_cluster_1.txt",
        "/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pf_cluster_2.txt",
        "/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pf_cluster_3.txt",
        "/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pf_cluster_4.txt",
        "/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pf_cluster_5.txt",
        "/Users/cheaulee/Desktop/hfproject/PF/Kmeans/pf_cluster_6.txt"
    ]

    # Load known drugs data
    known_genes, known_drugs_df = load_known_drugs(known_drugs_file_path)
    print(f"Loaded {len(known_genes)} known genes from {known_drugs_file_path}")

    # Prepare a list to store the results
    results = []
    known_genes_in_clusters = []

    # Iterate through each cluster file
    for cluster_file in cluster_files:
        # Load the cluster data
        with open(cluster_file, 'r') as file:
            cluster_genes = [line.strip() for line in file.readlines()]
        
        # Convert to a DataFrame
        cluster_genes_df = pd.DataFrame(cluster_genes, columns=['Gene_Name'])
        
        # Calculate the total number of genes and the number of known genes in the cluster
        total_genes = len(cluster_genes_df)
        filtered_genes_count = cluster_genes_df['Gene_Name'].isin(known_genes).sum()
        
        # Get the cluster number from the file name
        cluster_number = cluster_file.split('_')[-1].split('.')[0]
        
        # Append the result to the list
        results.append({
            "Cluster": cluster_number,
            "Total Number of Genes": total_genes,
            "Total Number of Known Genes": filtered_genes_count
        })
        
        # Find known genes in the current cluster and store them
        known_genes_in_cluster = cluster_genes_df[cluster_genes_df['Gene_Name'].isin(known_genes)].copy()
        known_genes_in_cluster['Cluster'] = cluster_number
        known_genes_in_clusters.append(known_genes_in_cluster)

    # Convert the results to a DataFrame and print
    results_df = pd.DataFrame(results)
    print(results_df)

    # Combine all known genes in clusters into a single DataFrame
    known_genes_in_clusters_df = pd.concat(known_genes_in_clusters)

    # Save the known genes in clusters to a new file
    known_genes_in_clusters_path = "/Users/cheaulee/Desktop/hfproject/PF/Kmeans/OT_clusters.csv"
    known_genes_in_clusters_df.to_csv(known_genes_in_clusters_path, index=False)
    
    print(f"Known genes in clusters saved to {known_genes_in_clusters_path}")

if __name__ == "__main__":
    main()






# Cluster pattern for druggability features 

def load_known_drugs(file_path):
    """Load known drugs dataset and extract unique gene symbols."""
    known_drugs_df = pd.read_csv(file_path, sep='\t')
    known_genes = set(known_drugs_df['symbol'].tolist())
    return known_genes, known_drugs_df

def main():
    # Paths to the input files
    known_drugs_file_path = "/Users/cheaulee/Desktop/hfproject/DATA/OpenTargets/EFO_0003144-known-drugs.tsv"
    cluster_files = [
        "/Users/cheaulee/Desktop/hfproject/DF/Kmeans/df_cluster_1.txt",
        "/Users/cheaulee/Desktop/hfproject/DF/Kmeans/df_cluster_2.txt",
        "/Users/cheaulee/Desktop/hfproject/DF/Kmeans/df_cluster_3.txt"
    ]

    # Load known drugs data
    known_genes, known_drugs_df = load_known_drugs(known_drugs_file_path)
    print(f"Loaded {len(known_genes)} known genes from {known_drugs_file_path}")

    # Prepare a list to store the results
    results = []
    known_genes_in_clusters = []

    # Iterate through each cluster file
    for cluster_file in cluster_files:
        # Load the cluster data
        with open(cluster_file, 'r') as file:
            cluster_genes = [line.strip() for line in file.readlines()]
        
        # Convert to a DataFrame
        cluster_genes_df = pd.DataFrame(cluster_genes, columns=['Gene_Name'])
        
        # Calculate the total number of genes and the number of known genes in the cluster
        total_genes = len(cluster_genes_df)
        filtered_genes_count = cluster_genes_df['Gene_Name'].isin(known_genes).sum()
        
        # Get the cluster number from the file name
        cluster_number = cluster_file.split('_')[-1].split('.')[0]
        
        # Append the result to the list
        results.append({
            "Cluster": cluster_number,
            "Total Number of Genes": total_genes,
            "Total Number of Known Genes": filtered_genes_count
        })
        
        # Find known genes in the current cluster and store them
        known_genes_in_cluster = cluster_genes_df[cluster_genes_df['Gene_Name'].isin(known_genes)].copy()
        known_genes_in_cluster['Cluster'] = cluster_number
        known_genes_in_clusters.append(known_genes_in_cluster)

    # Convert the results to a DataFrame and print
    results_df = pd.DataFrame(results)
    print(results_df)

    # Combine all known genes in clusters into a single DataFrame
    known_genes_in_clusters_df = pd.concat(known_genes_in_clusters)

    # Save the known genes in clusters to a new file
    known_genes_in_clusters_path = "/Users/cheaulee/Desktop/hfproject/DF/OT_clusters.csv"
    known_genes_in_clusters_df.to_csv(known_genes_in_clusters_path, index=False)
    
    print(f"Known genes in clusters saved to {known_genes_in_clusters_path}")

if __name__ == "__main__":
    main()





# Druggability features - overlapping genes between DrugnomeAI and Open Targets (known gene-drug interactions)

def load_common_genes(file_path):
    """Load common genes from the provided file."""
    common_genes_df = pd.read_csv(file_path)
    common_genes = set(common_genes_df['symbol'].tolist())
    return common_genes

def main():
    # Paths to the input files
    common_genes_file_path = "/Users/cheaulee/Desktop/hfproject/DF/overlapping_OT_DrugnomeAI.csv"
    cluster_files = [
        "/Users/cheaulee/Desktop/hfproject/DF/Kmeans/df_cluster_1.txt",
        "/Users/cheaulee/Desktop/hfproject/DF/Kmeans/df_cluster_2.txt",
        "/Users/cheaulee/Desktop/hfproject/DF/Kmeans/df_cluster_3.txt"
    ]

    # Load common genes data
    common_genes = load_common_genes(common_genes_file_path)
    print(f"Loaded {len(common_genes)} common genes from {common_genes_file_path}")

    # Prepare a list to store the results
    results = []
    common_genes_in_clusters = []

    # Iterate through each cluster file
    for cluster_file in cluster_files:
        # Load the cluster data
        with open(cluster_file, 'r') as file:
            cluster_genes = [line.strip() for line in file.readlines()]
        
        # Convert to a DataFrame
        cluster_genes_df = pd.DataFrame(cluster_genes, columns=['Gene_Name'])
        
        # Calculate the total number of genes and the number of common genes in the cluster
        total_genes = len(cluster_genes_df)
        filtered_genes_count = cluster_genes_df['Gene_Name'].isin(common_genes).sum()
        
        # Get the cluster number from the file name
        cluster_number = cluster_file.split('_')[-1].split('.')[0]
        
        # Append the result to the list
        results.append({
            "Cluster": cluster_number,
            "Total Number of Genes": total_genes,
            "Total Number of Common Genes": filtered_genes_count
        })
        
        # Find common genes in the current cluster and store them
        common_genes_in_cluster = cluster_genes_df[cluster_genes_df['Gene_Name'].isin(common_genes)].copy()
        common_genes_in_cluster['Cluster'] = cluster_number
        common_genes_in_clusters.append(common_genes_in_cluster)

    # Convert the results to a DataFrame and print
    results_df = pd.DataFrame(results)
    print(results_df)

    # Combine all common genes in clusters into a single DataFrame
    common_genes_in_clusters_df = pd.concat(common_genes_in_clusters)

    # Print the common genes in clusters
    print("Common genes in clusters:")
    print(common_genes_in_clusters_df)

if __name__ == "__main__":
    main()



