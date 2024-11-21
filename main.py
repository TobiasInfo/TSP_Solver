import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from TSP_generator import generate_tsp_instance
from TSP_solver import TSPConstraintGenerationSolver

def visualize_tsp_problem_and_solution(distance_matrix, coordinates, tour, total_distance):
    """
    Visualize both the initial complete graph and the TSP solution
    
    Args:
    - distance_matrix (np.ndarray): Distance matrix between cities
    - coordinates (np.ndarray): Coordinates of cities
    - tour (list): Optimal tour order
    - total_distance (float): Total tour distance
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    G_initial = nx.complete_graph(len(coordinates))
    
    for (u, v, d) in G_initial.edges(data=True):
        d['weight'] = distance_matrix[u][v]
    
    ax1.set_title("Initial Complete Graph")
    nx.draw_networkx_nodes(G_initial, pos=dict(zip(range(len(coordinates)), coordinates)), 
                            node_color='lightblue', node_size=300, ax=ax1)
    nx.draw_networkx_labels(G_initial, pos=dict(zip(range(len(coordinates)), coordinates)), ax=ax1)
    nx.draw_networkx_edges(G_initial, pos=dict(zip(range(len(coordinates)), coordinates)), 
                            width=0.5, alpha=0.5, ax=ax1)
    edge_labels = {(u,v): f'{distance_matrix[u][v]:.1f}' for (u,v) in G_initial.edges()}
    nx.draw_networkx_edge_labels(G_initial, pos=dict(zip(range(len(coordinates)), coordinates)), 
                                  edge_labels=edge_labels, ax=ax1)
    ax1.axis('off')    
    G_solution = nx.DiGraph()
    for i in range(len(coordinates)):
        G_solution.add_node(i, pos=coordinates[i])
    
    for i in range(len(tour)):
        current = tour[i]
        next_city = tour[(i+1) % len(tour)]
        G_solution.add_edge(current, next_city, weight=distance_matrix[current][next_city])
    
    ax2.set_title(f"Optimal TSP Tour\nTotal Distance: {total_distance:.2f}")
    nx.draw_networkx_nodes(G_solution, pos=dict(zip(range(len(coordinates)), coordinates)), 
                            node_color='lightblue', node_size=300, ax=ax2)
    nx.draw_networkx_labels(G_solution, pos=dict(zip(range(len(coordinates)), coordinates)), ax=ax2)
    nx.draw_networkx_edges(G_solution, pos=dict(zip(range(len(coordinates)), coordinates)), 
                            edgelist=list(G_solution.edges()), 
                            edge_color='red', 
                            arrows=True,
                            width=2,
                            connectionstyle='arc3,rad=0.1',
                            ax=ax2)
    
    solution_edge_labels = {(u,v): f'{distance_matrix[u][v]:.1f}' for (u,v) in G_solution.edges()}
    nx.draw_networkx_edge_labels(G_solution, pos=dict(zip(range(len(coordinates)), coordinates)), 
                                  edge_labels=solution_edge_labels, ax=ax2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    while True:
        try:
            num_cities = int(input("Enter the number of cities (minimum 3): "))
            if num_cities >= 3:
                break
            print("Number of cities must be at least 3.")
        except ValueError:
            print("Please enter a valid integer.")

    use_seed = input("Do you want to use a specific random seed? (yes/no): ").lower() == 'yes'
    
    if use_seed:
        while True:
            try:
                random_seed = int(input("Enter the random seed: "))
                break
            except ValueError:
                print("Please enter a valid integer.")
    else:
        random_seed = None

    while True:
        try:
            distance_type = int(input("Choose distance type (1.euclidean/2.random): "))

            if distance_type == 1:
                distance_type = 'euclidean'
                break
            elif distance_type == 2:
                distance_type = 'random'
                break
            else:
                print("Please enter a valid integer (1 or 2).")
        except ValueError:
            print("Invalid input. Please enter an integer.")


    distance_matrix, coordinates = generate_tsp_instance(
        num_cities=num_cities,
        seed=random_seed,
        distance_type=distance_type
    )
    print(distance_matrix)
    
    # Set some distances to np.inf to represent no direct connection
    # distance_matrix[0][1] = np.inf
    # distance_matrix[1][0] = np.inf
    # distance_matrix[2][3] = np.inf
    # distance_matrix[3][2] = np.inf
    
    solver = TSPConstraintGenerationSolver(distance_matrix)
    optimal_tour, total_distance = solver.solve()
    
    print("Optimal Tour:", optimal_tour)
    print("Total Distance:", total_distance)
    
    visualize_tsp_problem_and_solution(distance_matrix, coordinates, optimal_tour, total_distance)
if __name__ == "__main__":
    main()