import numpy as np
from scipy.optimize import minimize


def cal_lr(e11, e12, e21, e22):
    def equations(variables):
        n1, n2, delta, sigma = variables

        '''
        e11 = 1
        e12 = 1
        e21 = 1.264
        e22 = 1
        e31 = 2.7679
        e32 = 1
        e41 = 1.1166
        e42 = 0
        e51 = 19.0185
        e52 = 0
        '''

        eq1 = n1 - (e11 * delta + e12 * sigma)
        eq2 = n2 - (e21 * delta + e22 * sigma)

        return [eq1, eq2]

    # Constraint for n1 to n5 values between 0.1 and 1
    def constraints(variables):
        n1, n2, delta, sigma = variables
        return [n1 - 0.1, 1 - n1, n2 - 0.1, 1 - n2]

    # Initial guess for variables
    initial_guess = [1.0, 1.0, 0.5, 0.5]

    # Perform optimization
    result = minimize(
        fun=lambda x: np.sum(np.square(equations(x))),
        x0=initial_guess,
        constraints={'type': 'ineq', 'fun': constraints}
    )

    # Extract optimized values
    optimized_variables = result.x
    n1, n2, delta, sigma = optimized_variables

    # Print the results
    print(f"n1: {n1 * 0.0001}")
    print(f"n2: {n2 * 0.0001}")

    print(f"Optimal δ: {delta}")
    print(f"Optimal σ: {sigma}")
    return n1, n2