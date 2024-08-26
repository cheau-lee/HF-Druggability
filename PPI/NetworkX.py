import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



# Main pipeline 

def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def create_graph(interactions_df, node1_col, node2_col, weight_col):
    G = nx.Graph(name='Protein Interaction Graph')
    interactions = np.array(interactions_df[[node1_col, node2_col, weight_col]])
    for interaction in interactions:
        a = interaction[0]  # protein a node
        b = interaction[1]  # protein b node
        w = float(interaction[2])  # score as weighted edge where high scores = low weight
        G.add_weighted_edges_from([(a, b, w)])  # add weighted edge to graph
    return G

def rescale(l, newmin, newmax):
    arr = list(l)
    if max(arr) == min(arr):
        return [newmin] * len(arr)
    return [(x - min(arr)) / (max(arr) - min(arr)) * (newmax - newmin) + newmin for x in arr]

def calculate_centrality_measures(G):
    betweenness_centrality = nx.betweenness_centrality(G)
    return betweenness_centrality

def save_centrality_measures(betweenness_centrality):
    betweenness_centrality_df = pd.DataFrame(list(betweenness_centrality.items()), columns=['Node', 'Betweenness Centrality'])
    betweenness_centrality_df.to_csv('/Users/cheaulee/Desktop/hfproject/DATA/Network/C1_betweenness_centrality.csv', index=False)

def detect_communities(G):
    communities = list(nx.community.greedy_modularity_communities(G))
    community_mapping = {node: idx + 1 for idx, community in enumerate(communities) for node in community}  # Start from 1
    community_dict = {idx + 1: list(community) for idx, community in enumerate(communities)}  # Start from 1
    return community_mapping, community_dict

def save_communities(community_dict):
    community_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in community_dict.items()]))
    community_df = community_df.transpose()
    community_df.index.name = 'Community'
    community_df.columns = [f'Gene_{i}' for i in range(community_df.shape[1])]
    community_df.to_csv('/Users/cheaulee/Desktop/hfproject/DATA/Network/C1_communities.csv', index=True)
    return community_df

def combine_results(G, betweenness_centrality, community_mapping):
    degree_distribution = dict(G.degree())
    
    combined_df = pd.DataFrame({'Node': list(betweenness_centrality.keys())})
    combined_df['Degree of Distribution'] = combined_df['Node'].map(degree_distribution)
    combined_df['Betweenness Centrality'] = combined_df['Node'].map(betweenness_centrality)
    combined_df['Community'] = combined_df['Node'].map(community_mapping)
    combined_df.to_csv('/Users/cheaulee/Desktop/hfproject/DATA/Network/C1_combined.csv', index=False)
    return combined_df

def plot_graph(G, betweenness_centrality):
    graph_colormap = plt.get_cmap('plasma', 12)

    # Node color varies with Degree
    c = rescale([G.degree(v) for v in G], 0.0, 0.9)
    c = [graph_colormap(i) for i in c]

    # Node size varies with betweenness centrality - map to range [300, 3000]
    s = rescale([v for v in betweenness_centrality.values()], 300, 3000)

    # Edge width shows 1-weight to convert cost back to strength of interaction
    ew = rescale([float(G[u][v]['weight']) for u, v in G.edges], 0.5, 3)

    # Edge color also shows weight
    ec = rescale([float(G[u][v]['weight']) for u, v in G.edges], 0.1, 1)
    ec = [graph_colormap(i) for i in ec]

    # Plot the detailed interaction graph
    pos = nx.spring_layout(G, k=0.5)  # Increase the k value for better separation
    plt.figure(figsize=(19, 9), facecolor='white')  # Change the background to white

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=ec, width=ew)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=c, node_size=s)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='darkgrey', font_weight='bold',
                            verticalalignment='center', horizontalalignment='center')  # White font inside the bubbles

    plt.axis('off')
    plt.show()

def print_summary(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")

def main():
    file_path = '/Users/cheaulee/Desktop/hfproject/STRINGDB_C1.tsv'
    interactions_df = load_data(file_path)
    print("Column names in the TSV file:", interactions_df.columns)

    node1_col = 'node1'
    node2_col = 'node2'
    weight_col = 'combined_score'

    G = create_graph(interactions_df, node1_col, node2_col, weight_col)

    betweenness_centrality = calculate_centrality_measures(G)
    save_centrality_measures(betweenness_centrality)

    community_mapping, community_dict = detect_communities(G)
    save_communities(community_dict)

    combined_df = combine_results(G, betweenness_centrality, community_mapping)
    print("Combined Centrality Measures and Communities:", combined_df.head())

    print_summary(G)
    
    plot_graph(G, betweenness_centrality)

if __name__ == "__main__":
    main()



# Calculate mean and sd 

# Load the data from the CSV file
file_path = '/Users/cheaulee/Desktop/hfproject/DATA/Network/C1_combined.csv'  
data = pd.read_csv(file_path)

# Calculate mean and standard deviation for Degree of Distribution and Betweenness Centrality
mean_degree = data['Degree of Distribution'].mean()
std_degree = data['Degree of Distribution'].std()

mean_betweenness = data['Betweenness Centrality'].mean()
std_betweenness = data['Betweenness Centrality'].std()

# Print the results
print(f'Mean Degree of Distribution: {mean_degree}')
print(f'Standard Deviation of Degree of Distribution: {std_degree}')
print(f'Mean Betweenness Centrality: {mean_betweenness}')
print(f'Standard Deviation of Betweenness Centrality: {std_betweenness}')
