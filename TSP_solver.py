import numpy as np
import pulp

class TSPConstraintGenerationSolver:
    def __init__(self, distance_matrix):
        """
        Initialize the TSP solver with a distance matrix

        Args:
        - distance_matrix (np.ndarray): Distance matrix for the TSP problem
        """
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)

    def find_subcircuit(self, x_solution):
        """
        Find a subcircuit in the current solution

        Args:
        - x_solution (np.ndarray): Binary solution matrix

        Returns:
        - set of nodes in the subcircuit, or None if no subcircuit exists
        """
        visited = [False] * self.n

        for start in range(self.n):
            if not visited[start]:
                current = start
                current_circuit = [current]
                visited[current] = True

                while True:
                    # Find next node in the circuit
                    next_node = None
                    for j in range(self.n):
                        # Try to find the next node in the circuit
                        if x_solution[current][j] > 0.5 and not visited[j]:
                            next_node = j
                            break

                    # If no next node or back to start, check circuit
                    if next_node is None:
                        # If we are back to the start, check if the circuit is complete
                        if x_solution[current][start] > 0.5:
                            # Complete circuit found
                            if len(current_circuit) < self.n:
                                return set(current_circuit)
                        break

                    current = next_node
                    current_circuit.append(current)
                    visited[current] = True

        return None

    def solve(self, time_limit=60):
        """
        Solve the TSP using constraint generation

        Args:
        - time_limit (int): Maximum solving time in seconds

        Returns:
        - Optimal tour and its total distance
        """
        # Initial Model Setup (min Sum(d_ij * x_ij))
        model = pulp.LpProblem("TSP", pulp.LpMinimize)

        # Create the variables (x_ij)
        # Binary variables indicating if we travel from city i to city j
        x = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.distance_matrix[i][j] < np.inf:
                    x[i,j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')

        # Setup the Objective function: (min Sum(d_ij * x_ij))
        model += pulp.lpSum(self.distance_matrix[i][j] * x[i,j]
                             for i in range(self.n) for j in range(self.n) if i != j and self.distance_matrix[i][j] < np.inf)

        # Degree constraints: Each city has exactly one incoming and one outgoing arc
        # (sum of x_ij for i != j) == 1 (6.1.b)
        # (sum of x_ji for i != j) == 1 (6.1.c)
        for i in range(self.n):
            model += pulp.lpSum(x[i,j] for j in range(self.n) if j != i and self.distance_matrix[i][j] < np.inf) == 1
            model += pulp.lpSum(x[j,i] for j in range(self.n) if j != i and self.distance_matrix[j][i] < np.inf) == 1

        # Constraint generation loop
        while True:
            # Solve the pb with Coin-or Branch and Cut
            model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))

            # Converts solver variables to a numpy solution matrix
            x_solution = np.zeros((self.n, self.n))
            for (i,j), var in x.items():
                x_solution[i][j] = var.varValue

            print(x_solution)
            # Check for subcircuit
            subcircuit = self.find_subcircuit(x_solution)

            # If no subcircuit => we have an optimal solution
            if subcircuit is None:
                break

            # if we have a subcircuit, add a constraint to avoid it
            subtour_nodes = list(subcircuit)
            model += pulp.lpSum(x[i,j] for i in subtour_nodes for j in subtour_nodes if i != j and self.distance_matrix[i][j] < np.inf) <= len(subtour_nodes) - 1

        # Reconstruct optimal tour and total distance
        tour = []
        current = 0
        visited = set()

        while len(tour) < self.n:
            tour.append(current)
            visited.add(current)
            found_next = False
            for j in range(self.n):
                if (current, j) in x and x[current,j].varValue > 0.5 and j not in visited:
                    current = j
                    found_next = True
                    break
            if not found_next:
                break

        total_distance = sum(self.distance_matrix[tour[i]][tour[(i+1)%self.n]] for i in range(self.n))

        return tour, total_distance

# Example usage
if __name__ == "__main__":
    from TSP_generator import generate_tsp_instance

    # Generate TSP instance
    distance_matrix, _ = generate_tsp_instance(num_cities=10, seed=42)

    # Modify the distance matrix to represent a non-fully connected graph
    # For example, set some distances to np.inf to represent no direct connection
    distance_matrix[0][1] = np.inf
    distance_matrix[1][0] = np.inf
    distance_matrix[2][3] = np.inf
    distance_matrix[3][2] = np.inf

    # Solve TSP
    solver = TSPConstraintGenerationSolver(distance_matrix)
    optimal_tour, total_distance = solver.solve()

    print("Optimal Tour:", optimal_tour)
    print("Total Distance:", total_distance)
