import numpy as np
from src.cec2017.functions import f2, f13

# Numebr of iterations: tmax=budget/mu
# mu = number of

BUDGET = 10000
DIMENSION = 10
LB, UB = -100, 100

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