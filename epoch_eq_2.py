import numpy as np
from scipy.optimize import least_squares


# Define the equations
def equations(variables, T_slowest, C):
    T_1, T_2, T_3, T_4, T_5 = variables

    equation_1 = (-6.258777939013056e-17 * T_slowest / T_1 + (
                -7.017200427216381e-17 * 0.1) + 0.08333333333000005 * 1) * 12 * T_1 - C
    equation_2 = (-6.258777939013056e-17 * T_slowest / T_2 + (
                -7.017200427216381e-17 * 0.118151260504201) + 0.08333333333000005 * 1) * 12 * T_2 - C
    equation_3 = (-6.258777939013056e-17 * T_slowest / T_3 + (
                -7.017200427216381e-17 * 0.182285714285714) + 0.08333333333000005 * 1) * 12 * T_3 - C
    equation_4 = (-6.258777939013056e-17 * T_slowest / T_4 + (
                -7.017200427216381e-17 * 0.109360902255639) + 0.08333333333000005 * 1) * 12 * T_4 - C
    equation_5 = (-6.258777939013056e-17 * T_slowest / T_5 + (
                -7.017200427216381e-17 * 1) + 0.08333333333000005 * 1) * 12 * T_5 - C

    return [equation_1, equation_2, equation_3, equation_4, equation_5]


# Initial guesses for variables
initial_guess = [12, 12, 12, 12, 12]

# Known values
T_slowest = 1  # You can adjust this value if needed
C = 1  # You can adjust this value if needed

# Solve the least squares problem
result = least_squares(equations, initial_guess, args=(T_slowest, C))

# Print the solutions
T_1, T_2, T_3, T_4, T_5 = result.x
print(f"T_1: {T_1}")
print(f"T_2: {T_2}")
print(f"T_3: {T_3}")
print(f"T_4: {T_4}")
print(f"T_5: {T_5}")
print(f"T_slowest: {T_slowest}")
print(f"C: {C}")

print("Done")