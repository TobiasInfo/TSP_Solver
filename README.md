# Traveling Salesman Problem (TSP) Solver

This project implements a solver for the Traveling Salesman Problem (TSP) using the **Column Generation** approach. The TSP is a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting point. This implementation includes functionality to generate TSP instances, solve the problem using constraint generation, and visualize the results.

## Features

- **TSP Instance Generator**: 
  - Generate fully connected graphs with random or Euclidean distances between cities.
  - Visualize the complete graph representation of the problem.
  
- **TSP Solver**:
  - Solve the TSP using the constraint generation method.
  - Detect and resolve subtours using custom constraints.
  - Outputs the optimal route and its total distance.

- **Visualization**:
  - Display the initial problem graph with cities and their connections.
  - Highlight the optimal TSP tour with the computed total distance.

## How to Use

### Prerequisites
- Python 3.8 or later
- Required libraries: `numpy`, `pulp`, `matplotlib`, `networkx`

Install the required packages with:
```bash
pip install numpy pulp matplotlib networkx
```

## Running the Project

To run the project:

1. Clone this repository.
2. Run the main.py script:
```bash
    python main.py
```
3. Follow the prompts to:
  - Enter the number of cities (minimum: 3).
  - Specify whether to use a random seed for reproducibility.
  - Choose the type of distance generation (Euclidean or Random).

The script will generate a TSP instance, solve it, and display the solution graphically.

## Example

1. Input:
```bash
Enter the number of cities (minimum 3): 5
Do you want to use a specific random seed? (yes/no): yes
Enter the random seed: 42
Choose distance type (1.euclidean/2.random): 1
```

2. Output:
```bash
    Optimal Tour: [0, 3, 2, 4, 1]
    Total Distance: 195.76
```
The optimal tour is displayed graphically alongside the original problem graph.

## Configuration

  - Number of Cities: Customize the number of cities in the instance.
  - Random Seed: Use a specific seed for reproducibility.
  - Distance Type:
       - `Euclidean`: Distances are calculated using city coordinates in a 2D plane.
       - `Random`: Distances are randomly generated.

## Project Structure

  - `TSP_generator.py`:
        - Functions to generate TSP instances with different distance configurations.
        - Includes a visualization function for the complete graph.

   - `TSP_solver.py`:
        - Implements the TSP solver using column generation.
        - Adds constraints dynamically to eliminate subtours.
        - Returns the optimal tour and its total distance.

    - `main.py`:
        - Interactive script to configure, solve, and visualize TSP instances.

## Visualization Examples

- **Complete Graph**: Displays the original problem with cities and their connections.

- **Optimal Tour**: Highlights the shortest path calculated by the solver.

## License

This project is licensed under the MIT License.

Happy optimizing!
