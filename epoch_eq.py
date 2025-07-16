import numpy as np
import random
import math

# Define the equations
equations = np.array([[12, 4.0452, 0],
                      [ 20.01852804,  2.022, 0],
                      [22.111, 1.4472, 0],
                      [15.513, 3.0144,0],
                      [23.121162, 1.4676,0]])

best_error = float('inf')
best_solution = None

for i in range(1000000):
    # Generate random values within the range of 8-16 for results
    results = np.array([random.uniform(8, 16) for i in range(5)])

    # Solve using least squares
    solution, residuals, rank, singular_values = np.linalg.lstsq(equations, results, rcond=None)

    # Check if the current solution has positive values
    if np.all(solution >= 0):
        # Calculate the overall error (sum of all five errors)
        overall_error = np.sum(np.abs(np.dot(equations, solution) - results))
        print("Overall error:", overall_error)

        # Check if the current solution gives the least error
        if overall_error < best_error:
            best_error = overall_error
            best_solution = solution

# Get the values of A, B, C for the best solution
A, B, C = best_solution

print("\n")
print("Best values of A, B, C (with least overall error):")
print("A =", round(A,3))
print("B =",round(B,3))
print("C =", round(C,3))
print("\n")
print("best_error:", best_error)
print("\n")
print("Best epoch values of E1, E2, E3, E4, E5:")
print("E1 =", math.ceil(results[0]))
print("E2 =", math.ceil(results[1]))
print("E3 =", math.ceil(results[2]))
print("E4 =", math.ceil(results[3]))
print("E5 =", math.ceil(results[4]))
'''


import numpy as np
import random

# Define the equations
equations = np.array([[12, 4.0452, 0],
                      [15.48, 3.0144, 0],
                      [19.92, 2.022, 0],
                      [15.48, 3.0144, 0],
                      [23.04, 1.4676, 0]])

best_error = float('inf')
best_solution = None

for i in range(100):
    # Generate random values within the range of 8-16 for results
    results = np.array([random.uniform(8, 16) for _ in range(5)])

    # Solve using least squares
    solution, residuals, rank, singular_values = np.linalg.lstsq(equations, results, rcond=None)

    # Check if the current solution has positive values
    if np.all(solution > 0):
        # Calculate the overall error (sum of all five errors)
        overall_error = np.sum(np.abs(np.dot(equations, solution) - results))
        print("Overall error:", overall_error)

        # Check if the current solution gives the least error
        if overall_error < best_error:
            best_error = overall_error
            best_solution = solution

# Check if a valid solution was found
if best_solution is not None:
    # Get the values of A, B, C for the best solution
    A, B, C = best_solution

    print("\n")
    print("Best values of A, B, C (with least overall error):")
    print("A =", round(A, 3))
    print("B =", round(B, 3))
    print("C =", round(C, 3))
    print("\n")
    print("Best overall error:", best_error)
else:
    print("No valid solution found within the given constraints.")

'''