import time

import numpy as np


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# Arithmetic Optimization Algorithm (AOA)
def AOA(pop, obj_func, lb, ub, max_iter):
    [pop_size, dim] = pop.shape
    # Initialize the population
    epsilon = 1e-6
    gamma = 0.1
    # Evaluate the objective function
    pop_val = np.apply_along_axis(obj_func, 1, pop)

    # Initialize the best solution
    best_idx = np.argmin(pop_val)
    best_sol = pop[best_idx]
    best_val = pop_val[best_idx]
    Convergence = np.zeros(max_iter)

    ct = time.time()
    # Main loop
    for t in range(max_iter):

        # Compute the buoyancy force
        pop_diff = np.expand_dims(pop - best_sol, axis=1)
        pop_dist = np.sqrt(np.sum(pop_diff ** 2, axis=2))
        pop_weight = np.exp(-gamma * pop_dist)
        buoy_force = np.sum(pop_weight[:, :, np.newaxis] * pop_diff, axis=0)

        # Compute the random diffusion
        diff = np.random.randn(pop_size, dim)

        # Compute the resultant force
        res_force = buoy_force + diff

        # Update the position and evaluate the objective function
        pop_new = pop + res_force
        pop_new = np.clip(pop_new, lb, ub)
        pop_new_val = np.apply_along_axis(obj_func, 1, pop_new)

        # Update the best solution
        best_idx = np.argmin(pop_new_val)
        if pop_new_val[best_idx] < best_val:
            best_sol = pop_new[best_idx]
            best_val = pop_new_val[best_idx]

        # Check for convergence
        if np.linalg.norm(res_force) < epsilon:
            break

        # Update the population
        pop = pop_new
        pop_val = pop_new_val
        Convergence[t] = best_val

    ct = time.time() - ct
    return best_val, Convergence, best_sol, ct
