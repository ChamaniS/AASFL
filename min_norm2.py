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
e11=1
e12=1
e21=1.5395
e22=1
e31=5.6369
e32=1
e41=1.3682
e42=1
e51=23.1518
e52=1

n_values = np.array([
    [e11, e12],
    [e21, e22],
    [e31,e32],
    [e41, e42],
    [e51, e52]
])

d=10
if e12<1 or e11>d:
    b1 = 0.1
else:
    b1 = 1
if e22<1 or e21>d:
    b2 = 0.1
else:
    b2 = 1
if e32<1 or e31>d:
    b3 = 0.1
else:
    b3 = 1
if e42<1 or e41>d:
    b4 = 0.1
else:
    b4 = 1
if e52<1 or e51>d:
    b5 = 0.1
else:
    b5 = 1

# Target n values bounded by 0.1 and 1
n_target = np.array([b1, b2, b3, b4, b5])

print(n_target)
# Bounds for δ and σ (assuming a reasonable range)
bounds = [(0, 5), (0, 5)]

# Initial guess for δ and σ
initial_guess = [0.5, 0.5]

# Optimization process
result = minimize(fun=lambda x: np.sum(np.square(equations(x, n_target, n_values))),x0=initial_guess, bounds=bounds)

# Extracting the optimal values
optimal_values = result.x
δ_optimal, σ_optimal = optimal_values

# Calculate the corresponding n values using the optimal δ and σ
n_values_optimal = [δ_optimal * n_values[i][0] + σ_optimal * n_values[i][1] for i in range(len(n_values))]

# Print the results
print("Solutions:")
print("Optimal δ:", δ_optimal)
print("Optimal σ:", σ_optimal)
print("\n")
#print("Corresponding n values:", n_values_optimal)
print("n1 =", 0.0001*n_values_optimal[0])
print("n2 =", 0.0001*n_values_optimal[1])
print("n3 =", 0.0001*n_values_optimal[2])
print("n4 =", 0.0001*n_values_optimal[3])
print("n5 =", 0.0001*n_values_optimal[4])