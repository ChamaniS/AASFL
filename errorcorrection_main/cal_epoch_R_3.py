import numpy as np


def cal_epoch(e11, e12, e13, e21, e22, e23):
    A = np.array([[e11, e12, e13],[ e21,  e22, e23]])

    # Define the range for E1, E2, E3, E4, and E5 (between 8 and 16)
    lower_bound = 8
    upper_bound = 16

    # Initialize variables to store the best minimum norm solution and the corresponding E values
    best_min_norm = np.inf
    best_E_values = None

    # Loop through all possible values of E1, E2, E3, E4, and E5 within the specified range
    for E1 in np.arange(lower_bound, upper_bound + 1):
        for E2 in np.arange(lower_bound, upper_bound + 1):
            # Right-hand side vector b based on the current E values
            b = np.array([E1, E2])

            # Compute the least-squares solution using np.linalg.lstsq
            x_min, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Calculate the norm of the solution
            min_norm = np.linalg.norm(x_min)

            # Check if the current solution has a smaller norm than the previous best
            if min_norm < best_min_norm:
                best_min_norm = min_norm
                best_E_values = (E1, E2)

    # The best_min_norm now contains the minimum norm value
    # The best_E_values tuple contains the corresponding E values for the minimum norm solution
    alpha_min, beta_min, gamma_min = np.linalg.lstsq(A, np.array(best_E_values), rcond=None)[0]

    print("Minimum norm solution:")
    print(f"α = {alpha_min}")
    print(f"β = {beta_min}")
    print(f"ϒ = {gamma_min}")
    print("E1 =", best_E_values[0])
    print("E2 =", best_E_values[1])
    return best_E_values
