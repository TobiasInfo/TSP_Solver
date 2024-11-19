import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_tsp_instance(num_cities, seed=None, distance_type='euclidean'):
    """
    Generate a TSP problem instance with distances between cities.
    
    Args:
    - num_cities (int): Number of cities in the problem
    - seed (int, optional): Random seed for reproducibility
    - distance_type (str): Type of distance generation 
      - 'random': Random distances between 1 and 100
      - 'euclidean': Generates coordinates and calculates Euclidean distances
    
    Returns:
    - distance_matrix (np.ndarray): Square matrix of distances between cities
    - coordinates (np.ndarray): City coordinates (if euclidean)
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if distance_type == 'random':
        distance_matrix = np.random.randint(1, 101, size=(num_cities, num_cities))
        
        # Set the diagonal to 0 to ensure that the distance between a city and itself is 0
        np.fill_diagonal(distance_matrix, 0)
        
        # Ensure symmetry
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Create coordinates for cities to visualize the graph
        # /!\ These coordinates are not used in the optimization problem and do not represent the distances
        coordinates = np.random.rand(num_cities, 2) * 100
    
    elif distance_type == 'euclidean':
        # Generate random coordinates for cities
        coordinates = np.random.rand(num_cities, 2) * 100
        
        # Compute the Euclidean distances between each city
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    else:
        raise ValueError("Invalid distance_type. Choose 'random' or 'euclidean'")
    
    return distance_matrix, coordinates

def visualize_tsp_graph(distance_matrix, coordinates):
    """
    Visualize the TSP graph with cities and distances.
    
    Args:
    - distance_matrix (np.ndarray): Distance matrix between cities
    - coordinates (np.ndarray): Coordinates of cities
    """
    # Create the graph that is fully connected
    G = nx.complete_graph(len(distance_matrix))
    
    # Add edge weights from distance matrix
    for (u, v, d) in G.edges(data=True):
        d['weight'] = distance_matrix[u][v]
    

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos=dict(zip(range(len(coordinates)), coordinates)), 
                            node_color='lightblue', node_size=300)
    nx.draw_networkx_labels(G, pos=dict(zip(range(len(coordinates)), coordinates)))
    nx.draw_networkx_edges(G, pos=dict(zip(range(len(coordinates)), coordinates)), 
                            width=0.5, alpha=0.5)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=dict(zip(range(len(coordinates)), coordinates)), 
                                  edge_labels=edge_labels)
    
    plt.title("Complete Graph Representing TSP Instance")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    num_cities = 3
    distance_matrix, coordinates = generate_tsp_instance(num_cities, distance_type='random')
    print(distance_matrix)
    visualize_tsp_graph(distance_matrix, coordinates)
  
# Example usage    
# [[ 0.  82.5 49. ]
#  [82.5  0.  54. ]
#  [49.  54.   0. ]]
# Here we have 3 cities with random distances between them.
# We have distance 82.5 between city 0 and city 1, 49 between city 0 and city 2, and 54 between city 1 and city 2.