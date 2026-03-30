import numpy as np
from src.cec2017.functions import f2, f13

# Global constraints
BUDGET = 10000
DIMENSION = 10
LB, UB = -100, 100
SIGMA = 3
mu_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

best_mu_sigma_variations = [0.1, 1, 2, 3, 5, 10, 30]
BUDGET_5X = 50000

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
    print(f"\nTesting function: {fun.__name__} - (sigma=3, budget=10k)")
    print(f"{'mu':>5} | {'Mean':>15} | {'Std':>15} | {'Best':>15} | {'Worst':>15} | {'Calls':>12}")
    print("-" * 92)

    mu_means = {}

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

        mu_means[mu] = mean_val

        # Print formatted results
        print(f"{mu:5d} | {mean_val:15.2f} | {std_val:15.2f} | {min_val:15.2f} | {max_val:15.2f} | {last_calls:12.2f}")

    best_mu = min(mu_means, key=mu_means.get)

    # Testing Sigma impact
    print(f"--- Starting Sigma Impact Analysis for {fun.__name__} with mu={best_mu} and budget={BUDGET} ---")
    print(f"{'sigma':>7} | {'Mean':>15} | {'Std':>15} | {'Best':>15} | {'Calls':>10}")
    print("-" * 75)

    sigma_means = {}
    for s in best_mu_sigma_variations:
        sig_results = []
        calls_sigma = 0

        for _ in range(25):
            res, calls_sigma = evolutionary_algorithm(fun, best_mu, s, BUDGET)
            sig_results.append(res)

        current_mean = np.mean(sig_results)
        sigma_means[s] = current_mean
        print(f"{s:7.1f} | {current_mean:15.2f} | {np.std(sig_results):15.2f} | {np.min(sig_results):15.2f} | {calls_sigma:10.0f}")

    # Testing budget impact
    print(f"--- Starting Budget Impact Analysis for {fun.__name__} with mu={best_mu}, sigma={SIGMA} and budget={BUDGET_5X} ---")

