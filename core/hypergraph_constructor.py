import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def construct_incidence_matrix_knn(features, n_neighbors=5):
    """
    Constructs an incidence matrix H for a hypergraph using K-Nearest Neighbors.
    Each node forms a hyperedge with its k-1 nearest neighbors.
    
    features: (num_nodes, feature_dim) tensor or numpy array
    return: H of shape (num_nodes, num_edges)
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
        
    num_nodes = features.shape[0]
    
    # We use scikit-learn for nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # In KNN based hypergraph, each node creates a hyperedge connecting itself and its neighbors
    num_edges = num_nodes
    H = np.zeros((num_nodes, num_edges))
    
    for edge_idx, neighbors in enumerate(indices):
        for node_idx in neighbors:
            H[node_idx, edge_idx] = 1.0
            
    return torch.tensor(H, dtype=torch.float32)

def construct_incidence_matrix_kmeans(features, n_clusters=10):
    """
    Constructs an incidence matrix H using KMeans clustering.
    Each cluster becomes a hyperedge.
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
        
    num_nodes = features.shape[0]
    
    # We use scikit-learn for KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(features)
    labels = kmeans.labels_
    
    num_edges = n_clusters
    H = np.zeros((num_nodes, num_edges))
    
    for node_idx, cluster_idx in enumerate(labels):
        H[node_idx, cluster_idx] = 1.0
        
    return torch.tensor(H, dtype=torch.float32)
