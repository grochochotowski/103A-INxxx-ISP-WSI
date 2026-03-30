import tkinter as tk
import numpy as np

from .a2_main import steepest_descent, visualize, booth
from src.cec2017.functions import f1, f2, f3

# ===================== RUN ALGORITHM =====================
def run_algorithm():
    beta_booth = float(beta_booth_entry.get())
    beta_cec = float(beta_cec_entry.get())
    max_iter = int(iter_entry.get())

    summary = []

    # Main loop through functions
    for fun in [booth, f1, f2, f3]:
        path_collection = []
        results = []
        best_val = float('inf')
        best_coords = None

        dim = 2 if fun == booth else 10
        beta = beta_booth if fun == booth else beta_cec

        # Loop in function for 25 test cases
        for _ in range(25):
            start_point = np.random.uniform(-100, 100, dim)
            optimum, path = steepest_descent(fun, start_point, beta, max_iter)
            path_collection.append(path)
            current_val = fun(optimum)
            results.append(current_val)
            if best_val > current_val:
                best_val = current_val
                best_coords = optimum

        # Adding results to summary
        summary.append({
            "name": fun.__name__,
            "best" : np.min(results),
            "worst": np.max(results),
            "mean": np.mean(results),
            "std": np.std(results),
            "best_coords": best_coords
        })

        # Drawing las run of each case
        for path in path_collection[:3]:
            path_2d = path[:, :2]
            visualize(fun, path_2d, 100)

    # Summary display
    print(f"{'Function':<10} | {'Best':<15} | {'Worst':<15} | {'Mean':<15} | {'Std Dev':<15} | {'Best'}")
    print("-" * 11 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 100)
    for row in summary:
        n = row['name']
        b = f"{row['best']:.4e}"
        w = f"{row['worst']:.4e}"
        m = f"{row['mean']:.4e}"
        s = f"{row['std']:.4e}"
        coords_list = [f"{x:.2f}" for x in row['best_coords']]
        c = "[" + "; ".join(coords_list) + "]"
        print(f"{n:<10} | {b:<15} | {w:<15} | {m:<15} | {s:<15} | {c}")

# ===================== GUI =====================
root = tk.Tk()
root.title("Steepest Descent")
root.option_add("*Font", "Arial 12")
main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack()

# Beta Booth
tk.Label(main_frame, text="Beta Booth").pack()
beta_booth_entry = tk.Entry(main_frame)
beta_booth_entry.insert(0, "0.01")
beta_booth_entry.pack(pady=5)

# Beta CEC
tk.Label(main_frame, text="Beta CEC 2017").pack()
beta_cec_entry = tk.Entry(main_frame)
beta_cec_entry.insert(0, "1e-9")
beta_cec_entry.pack(pady=5)

# Max iterations
tk.Label(main_frame, text="Iterations").pack()
iter_entry = tk.Entry(main_frame)
iter_entry.insert(0, "1000")
iter_entry.pack(pady=5)

# Run
tk.Button(main_frame, text="Run", command=run_algorithm).pack(pady=5)
root.mainloop()
