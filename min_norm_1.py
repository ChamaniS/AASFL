import numpy as np
from scipy.optimize import minimize

def equations(params):
    δ, σ = params[0], params[1]
    n1 = 1*δ + 1*σ
    n2 = 1.5395*δ + 1*σ
    n3 = 5.6369*δ + 1*σ
    n4 = 1.3682*δ + 1*σ
    n5 = 23.1518*δ + 1*σ

    return [n1, n2, n3 , n4, n5]

# Bounds for δ and σ (assuming a reasonable range)
bounds = [(0, None), (0, None)]

# Initial guess for δ and σ
initial_guess = [0.5, 0.5]

# Optimization process
result = minimize(lambda x: np.sum(np.array(equations(x))**2), initial_guess, bounds=bounds)

# Extracting the optimal values
optimal_values = result.x
delta_optimal, sigma_optimal = optimal_values

# Calculate the corresponding n values using the optimal δ and σ
n1_optimal = 1 * delta_optimal + 1 * sigma_optimal
n2_optimal = 1.5395 * delta_optimal + 1 * sigma_optimal
n3_optimal =  5.6369 * delta_optimal + 1 * sigma_optimal
n4_optimal = 1.3682 * delta_optimal + 1 * sigma_optimal
n5_optimal = 23.1518 * delta_optimal + 1 * sigma_optimal

# Print the results
print("Optimal δ:", delta_optimal)
print("Optimal σ:", sigma_optimal)
print("Corresponding n1:", n1_optimal)
print("Corresponding n2:", n2_optimal)
print("Corresponding n3:", n3_optimal)
print("Corresponding n4:", n4_optimal)
print("Corresponding n5:", n5_optimal)
