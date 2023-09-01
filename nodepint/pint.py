#%%
## Time-parallel strategies for nodepint

import jax
import jax.numpy as jnp
from functools import partial

import numpy as np


def select_root_finding_function(pint_scheme:str):

    if pint_scheme=="newton":
        root_finder = newton_root_finder
    elif pint_scheme=="direct":
        root_finder = direct_root_finder       ## Use partial if necessary
    elif pint_scheme=="sequential":
        root_finder = sequential_root_finder
    else:
        raise ValueError("Unknown time-parallel scheme")

    return root_finder


def shooting_function(Z, z0, nb_splits, times, rhs, integrator):

    ## Split times among N = nb_processors
    times = np.array(times)[:, jnp.newaxis]
    split_times = np.array_split(times, nb_splits)

    Z_ = [z0]
    nz = z0.shape[0]

    for n in range(nb_splits):      ## TODO do this in parallel   
        ts = split_times[n]
        if n < nb_splits-1:
            ts = np.concatenate([ts, split_times[n+1][0, jnp.newaxis]])      ## TODO do this just once, and reuse the ts later on
        ts = tuple(ts.flatten())

        z_next = integrator(rhs, Z[n*nz:(n+1)*nz], t=ts)[-1,...]
        Z_.append(z_next)

    return Z - jnp.concatenate(Z_, axis=0)


#%%

## Newton's method - TODO we still need to define:
# - custom JVP rule https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
# - example of fixed-point interation in PyTorch http://implicit-layers-tutorial.org/deep_equilibrium_models/
@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def newton_root_finder(func, B0, z0, nb_splits, times, rhs, integrator, learning_rate, tol, max_iter):
    grad = jax.jacfwd(func)

    B = B0

    for k in range(max_iter):

        ## This also tells us whether we are recompiling
        print("PinT iteration counter: ", k)

        grad_inv = jnp.linalg.inv(grad(B, z0, nb_splits, times, rhs, integrator))
        func_eval = func(B, z0, nb_splits, times, rhs, integrator)

        B_new = B - learning_rate * grad_inv @ func_eval

        # if jnp.linalg.norm(B_new - B) < tol:  ## TODO not jit-friendly
        #     break

        B = B_new

    return B



@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def direct_root_finder(func, B0, z0, nb_splits, times, rhs, integrator, learning_rate=1., tol=1e-6, maxiter=10):
    grad = jax.jacfwd(func)

    B = B0

    for _ in range(maxiter):
        ## Solve a linear system
        grad_eval = grad(B, z0, nb_splits, times, rhs, integrator)
        func_eval = func(B, z0, nb_splits, times, rhs, integrator)

        B_new = B + jnp.linalg.solve(grad_eval, -func_eval)

        if jnp.linalg.norm(B_new - B) < tol:
            break
        B = B_new

    return B



def sequential_root_finder(func, z0, B0):
    pass
