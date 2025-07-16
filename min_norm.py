import numpy as np
from scipy.optimize import minimize

def objective_function(x, *args):
    n_values = args[0]
    target_bounds = args[1]
    n_target = args[2]

    n_calculated = np.sum(x[:, None] * n_values, axis=0)

    # Calculate the differences between calculated n values and target bounds
    differences = n_target - n_calculated
    weighted_differences = differences * target_bounds

    # Minimize the sum of squared weighted differences
    return np.sum(weighted_differences**2)

# Coefficients of δ and σ for each equation
coefficients = np.array([
    [1, 1],
    [1.264, 1],
    [2.7679, 1],
    [1.1166, 0],
    [19.0185, 0]
])

# Target n values bounded by 0.1 and 1
n_target = np.array([1, 1, 1, 0.1, 0.1])

# Target bounds (weights for differences calculation)
target_bounds = np.array([0.1, 0.1, 0.1, 0.9, 0.9])

# Constraints on σ coefficients
sigma_constraints = {'type': 'eq', 'fun': lambda x: x[1]}

# Bounds for δ and σ (assuming a reasonable range)
bounds = [(0, None), (0, None)]

# Initial guess for δ and σ
initial_guess = [1.0, 1.0]

# Optimization process
result = minimize(objective_function, initial_guess, args=(coefficients.T, target_bounds, n_target), bounds=bounds, constraints=sigma_constraints)

# Extracting the optimal values
optimal_values = result.x
delta_optimal, sigma_optimal = optimal_values

# Calculate the corresponding n values using the optimal δ and σ
n_values_optimal = delta_optimal * coefficients[:, 0] + sigma_optimal * coefficients[:, 1]

# Print the results
print("Optimal δ:", delta_optimal)
print("Optimal σ:", sigma_optimal)
print("Corresponding n values:", n_values_optimal)
