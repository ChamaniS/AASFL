import numpy as np
from scipy.optimize import minimize

def equations(variables, *args):
    δ, σ = variables
    n_target = args[0]
    n_values = args[1]

    n_calculated = [δ * n_values[i][0] + σ * n_values[i][1] for i in range(len(n_values))]

    # Constraints on n values
    constraints = [n_target[i] - n_calculated[i] for i in range(len(n_target))]

    return constraints

# Coefficients of δ and σ for each equation
n_values = np.array([
    [1, 1],
    [1.264, 1],
    [2.7679, 1],
    [1.1166, 0],
    [19.0185, 0]
])

# Target n values bounded by 0.1 and 1
n_target = np.array([0.1, 0.1, 0.1, 1, 1])

# Bounds for δ and σ (assuming a reasonable range)
bounds = [(0, None), (0, None)]

# Initial guess for δ and σ
initial_guess = [0.5, 0.5]

# Optimization process
result = minimize(fun=lambda x: np.sum(np.square(equations(x, n_target, n_values))),
                  x0=initial_guess, bounds=bounds)

# Extracting the optimal values
optimal_values = result.x
δ_optimal, σ_optimal = optimal_values

# Calculate the corresponding n values using the optimal δ and σ
n_values_optimal = [δ_optimal * n_values[i][0] + σ_optimal * n_values[i][1] for i in range(len(n_values))]

# Print the results
print("Optimal δ:", δ_optimal)
print("Optimal σ:", σ_optimal)
print("Corresponding n values:", n_values_optimal)
