import numpy as np
from src.cec2017.functions import f2, f13

# Global constraints
BUDGET = 10000
DIMENSION = 10
LB, UB = -100, 100
SIGMA = 3
mu_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

def evolutionary_algorithm(fun, mu, sigma, budget):
    calls = mu

    # Initialize population
    population = np.random.uniform(LB, UB, (mu, DIMENSION))
    fitness = np.array([fun(ind) for ind in population])

    # Calculate iterations
    t_max = (budget // mu) - 1

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
        calls += mu

    return np.min(fitness), calls

# Main loop through functions
for fun in [f2, f13]:
    print(f"\nTesting function: {fun.__name__}")
    print(f"{'mu':>5} | {'Mean':>15} | {'Std':>15} | {'Best':>15} | {'Worst':>15} | {'Calls':>12}")
    print("-" * 92)

    for mu in mu_values:
        results = []
        last_calls = 0

        for _ in range(25):
            final_best, total_calls = evolutionary_algorithm(fun, mu, SIGMA, BUDGET)
            results.append(final_best)
            last_calls = total_calls

        # Calculate stats
        mean_val = np.mean(results)
        std_val = np.std(results)
        min_val = np.min(results)
        max_val = np.max(results)

        # Print formatted results
        print(f"{mu:5d} | {mean_val:15.2f} | {std_val:15.2f} | {min_val:15.2f} | {max_val:15.2f} | {last_calls:12.2f}")