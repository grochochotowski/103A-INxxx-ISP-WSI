import numpy as np
from src.cec2017.functions import f2, f13

# Global constraints
BUDGET = 10000
DIMENSION = 10
LB, UB = -100, 100

def evolutionary_algorithm(fun, mu, sigma, budget):
    # Initialize population
    population = np.random.uniform(LB, UB, (mu, DIMENSION))
    fitness = np.array([fun(ind) for ind in population])

    # Calculate iterations
    t_max = budget // mu

    for t in range(t_max):
        new_population = []
        for _ in range(mu):
            idx1, idx2 = np.random.choice(mu, 2, replace=False)

            # Choose parent
            if fitness[idx1] < fitness[idx2]:
                parent = population[idx1]
            else:
                parent = population[idx2]

            # Create new child from the parent
            child = parent + np.random.normal(0, sigma, DIMENSION)
            child = np.clip(child, LB, UB)
            new_population.append(child)

        population = np.array(new_population)
        fitness = np.array([fun(ind) for ind in population])
    return np.min(fitness)

# Main loop through functions
for fun in [f2, f13]:
    path_collection = []
    results = []
    best_val = float('inf')
    best_coords = None

    # Loop in function for 25 test cases
    for _ in range(25):
        ... = np.random.uniform(-100, 100, 10)
        optimum, path = ...
        path_collection.append(path)
        current_val = fun(optimum)
        results.append(current_val)
        if best_val > current_val:
            best_val = current_val
            best_coords = optimum