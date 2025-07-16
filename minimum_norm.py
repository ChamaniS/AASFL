import numpy as np

# Coefficient matrix A
A = np.array([
    [1, 1],
    [1.264, 1],
    [2.7679, 1],
    [1.1166, 0.1],
    [19.0185, 0.1]
])

# Define the range for E1, E2, E3, E4, and E5 (between 8 and 16)
lower_bound = 0.1
upper_bound = 1

# Initialize variables to store the best minimum norm solution and the corresponding E values
best_min_norm = np.inf
best_E_values = None

# Loop through all possible values of E1, E2, E3, E4, and E5 within the specified range
for n1 in np.arange(lower_bound, upper_bound):
    for n2 in np.arange(lower_bound, upper_bound):
        for n3 in np.arange(lower_bound, upper_bound):
            for n4 in np.arange(lower_bound, upper_bound):
                for n5 in np.arange(lower_bound, upper_bound):
                    # Right-hand side vector b based on the current E values
                    b = np.array([n1, n2, n3, n4, n5])

                    # Compute the least-squares solution using np.linalg.lstsq
                    x_min, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                    # Calculate the norm of the solution
                    min_norm = np.linalg.norm(x_min)

                    # Check if the current solution has a smaller norm than the previous best
                    if min_norm < best_min_norm:
                        best_min_norm = min_norm
                        best_n_values = (n1, n2, n3, n4, n5)

# The best_min_norm now contains the minimum norm value
# The best_E_values tuple contains the corresponding E values for the minimum norm solution
delta_min, sigma_min = np.linalg.lstsq(A, np.array(best_n_values), rcond=None)[0]

print("Minimum norm solution:")
print(f"δ = {delta_min}")
print(f"σ = {sigma_min}")
print("Corresponding n values:", best_n_values)
