import itertools
import numpy as np
import time

m = np.array([8, 3, 5, 2]) #masa przedmiotów
M = np.sum(m)/2 #niech maksymalna masa plecaka będzie równa połowie masy przedmiotów
p = np.array([16, 8, 9, 6]) #wartość przedmiotów

if len(m) != len(p):
    raise ValueError('m and p must have same length')

def backpack_exhaustive(m, M, p):
    start = time.process_time()
    n = len(m)
    best_value = 0
    best_combination = ()
    for r in range(n+1):
        for combination in itertools.combinations(range(n), r):
            total_mass = sum(m[i] for i in combination)
            total_value = sum(p[i] for i in combination)
            if total_value >= best_value and total_mass <= M:
                best_value = total_value
                best_combination = combination
    end = time.process_time()
    total_time = end - start
    return np.array(best_combination), best_value, total_time

def backpack_heuristic(m, M, p):
    start = time.process_time()
    ratio = p/m
    sorted_i = np.argsort(-ratio)
    mass = 0
    value = 0
    combination = []
    for i in sorted_i:
        if mass + m[i] <= M:
            combination.append(i.item())
            mass += m[i]
            value += p[i]
    end = time.process_time()
    total_time = end - start
    return combination, value, total_time

exhaustive_combination, exhaustive_value, exhaustive_time = backpack_exhaustive(m, M, p)
heuristic_combination, heuristic_value, heuristic_time = backpack_heuristic(m, M, p)

print("\nSize of a backpack: ", M)

print("\nExhaustive solution:")
print("Items: ", exhaustive_combination)
print("Value: ", exhaustive_value)
time_format_ex = "{0:02f}s".format(exhaustive_time).replace(".", ",")
print(f"Time: {time_format_ex}")

print("\nHeuristic solution:")
print("Items:", heuristic_combination)
print("Value:", heuristic_value)
time_format_he = "{0:02f}s".format(heuristic_time).replace(".", ",")
print(f"Time: {time_format_he}")

# ====== MINUTE TEST ======
def minute_test():
    n = 5
    exhaustive_time_test = 0
    print("\nMinute test:")
    print(f"{'n':>3} | {'Exhaustive [s]':>15} | {'Heuristic [s]':>15}")
    print("-" * 4 + "|" + "-" * 17 + "|" + "-" * 17)
    # Minute test loop (increase n by 1 every test, until time higher than 1 minute)
    while exhaustive_time_test < 60:
        random_mass = np.random.randint(1, 10, size=n)
        random_value = np.random.randint(1, 20, size=n)
        backpack_mass = np.sum(random_mass)/2

        _, _, exhaustive_time_test  = backpack_exhaustive(random_mass, backpack_mass, random_value)
        _, _, heuristic_time_test = backpack_heuristic(random_mass, backpack_mass, random_value)
        format_ex = f"{exhaustive_time_test:.8f}".replace(".", ",")
        format_he = f"{heuristic_time_test:.8f}".replace(".", ",")
        print(f"{n:>3} | {format_ex:>15} |  {format_he:>15}")
        n+=1

minute_test()
