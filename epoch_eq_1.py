import numpy as np
from scipy.optimize import minimize
import math

'''
def objective(x):
    a, b, c = x
    e1 = 12*a + 4.0452*b + 0*c
    e2 = 20.01852804*a + 2.022*b + 0*c
    e3 = 22.111*a + 1.4472*b + 0*c
    e4 = 15.513*a + 3.0144*b + 0*c
    e5 = 23.121162*a + 1.4676*b + 0*c

    error = abs(e1 - target_e1) + abs(e2 - target_e2) + abs(e3 - target_e3) + abs(e4 - target_e4) + abs(e5 - target_e5)
    return error

target_e1 = np.random.uniform(8, 16)  # Set the target values for E1 to vary between 8 and 16
target_e2 = np.random.uniform(8, 16)  # Set the target values for E2 to vary between 8 and 16
target_e3 = np.random.uniform(8, 16)  # Set the target values for E3 to vary between 8 and 16
target_e4 = np.random.uniform(8, 16)  # Set the target values for E4 to vary between 8 and 16
target_e5 = np.random.uniform(8, 16)  # Set the target values for E5 to vary between 8 and 16

best_error = float('inf')
best_solution = None

for i in range(100):
    # Random initial guess for the variables α, β, and ϒ
    x0 = np.random.uniform(0, 1, size=3)

    # Bounds for the variables to restrict them to positive values
    bounds = [(0, None), (0, None), (0, None)]

    # Solve the optimization problem
    result = minimize(objective, x0, bounds=bounds)

    # Check if the current solution has the lowest error so far
    if result.success and result.fun < best_error:
        best_error = result.fun
        print("best_error:", best_error)
        best_solution = result.x

# Extract the best values for α, β, and ϒ
best_alpha, best_beta, best_gamma = best_solution
'''


import numpy as np

def least_squares_solution(A, b):
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x

# Define the coefficients of the equations
equations_coefficients = np.array([
    [12, 4.0452, 0],
    [20.01852804, 2.022, 0],
    [22.111, 1.4472, 0],
    [15.513, 3.0144, 0],
    [23.121162, 1.4676, 0]
])

best_error = float('inf')
best_solution = None

for _ in range(100):
    # Generate random target values within the range of 8 to 16
    target_values = np.random.uniform(8, 16, size=(5,))

    # Solve the least squares problem to find the solution
    solution = least_squares_solution(equations_coefficients, target_values)

    # Check if the current solution has the lowest error so far
    error = np.sum(np.abs(np.dot(equations_coefficients, solution) - target_values))
    if error < best_error:
        best_error = error
        print("best_error:", best_error)
        best_solution = solution

# Extract the best values for α, β, and ϒ
best_alpha, best_beta, best_gamma = best_solution

print("Best solution:")
print(f"Alpha: {best_alpha}")
print(f"Beta: {best_beta}")
print(f"Gamma: {best_gamma}")
print(f"Overall error: {best_error}")

print("\n")
print("Best values of A, B, C (with least overall error):")
print("A =", round(best_alpha,3))
print("B =",round(best_beta,3))
print("C =", round(best_gamma,3))
print("\n")
print("best_error:", best_error)
print("\n")
print("Best epoch values of E1, E2, E3, E4, E5:")
print("E1 =", math.ceil(12*best_alpha + 4.0452*best_beta + 0*best_gamma))
print("E2 =", math.ceil(20.01852804*best_alpha + 2.022*best_beta + 0*best_gamma))
print("E3 =", math.ceil(22.111*best_alpha + 1.4472*best_beta + 0*best_gamma))
print("E4 =", math.ceil(15.513*best_alpha + 3.0144*best_beta + 0*best_gamma))
print("E5 =", math.ceil(23.121162*best_alpha + 1.4676*best_beta + 0*best_gamma))
