import numpy as np
import matplotlib.pyplot as plt

# ===================== BOOTH =====================
# xx = [x1, x2]
# x1 = xx(1);
# x2 = xx(2);
# term1 = (x1 + 2*x2 - 7)^2;
# term2 = (2*x1 + x2 - 5)^2;
# y = term1 + term2;

def booth(x):
    x1, x2 = x
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

# ===================== NUMERICALLY CALCULATED GRADIENT =====================
# θnew = θold − β∇f(θold)
# x_{k+1} = x_k - β∇f(x_k)

# ∇f(x_k) >>> f{prim} = (f(x_next) - f(x_prev)) / 2h
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_next  = x.copy()
        x_prev  = x.copy()
        x_next[i] += h
        x_prev[i] -= h
        grad[i] = (f(x_next) - f(x_prev)) / (2*h)

    return grad

# x_{k+1} = x_k - β * grad
def steepest_descent(f, x0, beta, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    points = [x.copy()]

    for i in range(max_iter):
        grad = numerical_gradient(f, x)
        x_new = np.clip(x - beta * grad, -100, 100) # Space cube constrains [-/+100]
        points.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(points)

# ===================== CREATE VISUALIZATION GRAPH =====================
def visualize(f, steps, max_x, plot_step=2):
    # create grid
    x_arr = np.arange(-max_x, max_x, plot_step)
    y_arr = np.arange(-max_x, max_x, plot_step)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.zeros_like(X)

    # Calculating dimensions for drawing a graph
    dim = steps.shape[1]

    # calculate functions
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = np.zeros(dim)
            v[0] = X[i, j]
            v[1] = Y[i, j]
            Z[i, j] = f(v)

    # draw contour map
    plt.contour(X, Y, Z, 20)

    # draw arrows
    for i in range(len(steps)-1):
        x_start = steps[i][0]
        y_start = steps[i][1]

        dx = steps[i+1][0] - steps[i][0]
        dy = steps[i+1][1] - steps[i][1]

        plt.arrow(x_start, y_start, dx, dy, head_width=0.3, head_length=0.3)

    # mark start and optimum
    plt.scatter(steps[0][0], steps[0][1], color="red", label="Start")
    plt.scatter(steps[-1][0], steps[-1][1], color="green", label="Optimum")

    # show plot
    plt.title(f.__name__)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
