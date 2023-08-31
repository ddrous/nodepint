#%%
## Time-parallel strategies for nodepint

import jax
import jax.numpy as jnp
from functools import partial

import numpy as np



# def solve_multiple_shooting(pint_scheme:str, B0, nb_processors, times, integrator):
#     # mtp_sht_func = define_shooting_function(nb_processors, times, integrator)
#     root_finder = select_root_finding_function(pint_scheme)

#     # return shooting_func(mtp_sht_func, x0, method='hybr', jac=None, tol=None, callback=None, options=None)
#     # return shooting_func(mtp_sht_func, B0)

#     return root_finder(shooting_function, B0)



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


# def define_shooting_function(nb_splits, times, rhs, integrator):
#     ## Split times among N = nb_processors
#     split_times = jnp.array_split(times, nb_splits)


#     ## Define the actual shooting function
#     def mtp_shooting_func(Z, z0):

#         # print("mm")
#         # print("All Arguments: ", Z, z0)

#         Z_ = [z0]
#         nz = z0.shape[0]

#         for n in range(nb_splits):      ## TODO do this in parallel   
#             ts = split_times[n]
#             if n < nb_splits-1:
#                 ts = jnp.concatenate([ts, split_times[n+1][0, jnp.newaxis]])      ## TODO do this just once, and reuse the ts later on

#             # z_next = integrator(rhs, Z[n], t=ts)[-1,...]
#             print("Types of all arguments: ", type(rhs), type(Z), type(ts), type(z0))
#             z_next = integrator(rhs, Z[n*nz:(n+1)*nz], t=ts)[-1,...]
#             Z_.append(z_next)

#         # return Z - jnp.stack(Z_, axis=0)
#         return Z - jnp.concatenate(Z_, axis=0)

#     return mtp_shooting_func


# @partial(jax.jit, static_argnums=(2, 3, 4, 5))
def shooting_function(Z, z0, nb_splits, times, rhs, integrator):
    ## Split times among N = nb_processors

    # times = jnp.array(times)[:, jnp.newaxis]
    # split_times = jnp.array_split(times, nb_splits)

    times = np.array(times)[:, jnp.newaxis]
    split_times = np.array_split(times, nb_splits)

    Z_ = [z0]
    nz = z0.shape[0]

    # print("Check every argument's type:", type(Z), type(z0), type(times), type(rhs), type(integrator))

    for n in range(nb_splits):      ## TODO do this in parallel   
        ts = split_times[n]
        if n < nb_splits-1:
            ts = np.concatenate([ts, split_times[n+1][0, jnp.newaxis]])      ## TODO do this just once, and reuse the ts later on
            ## CHCEKS NANs in ts
        ts = tuple(ts.flatten())
        # ts = tuple(map(lambda x: float(x), ts))

        # z_next = integrator(rhs, Z[n], t=ts)[-1,...]
        # print("Types of all arguments: ", type(rhs), type(Z), type(ts), type(z0))
        # print("SHapes of every element:", Z.shape, z0.shape, len(ts))

        # print("Currently processor number:", n)
        # print("Current size of Z:", Z.shape)

        z_next = integrator(rhs, Z[n*nz:(n+1)*nz], t=ts)[-1,...]
        Z_.append(z_next)

    # return Z - jnp.stack(Z_, axis=0)
    return Z - jnp.concatenate(Z_, axis=0)


#%%

## Newton's method - TODO we still need to define:
# - custom JVP rule https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
# - example of fixed-point interation in PyTorch http://implicit-layers-tutorial.org/deep_equilibrium_models/
@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def newton_root_finder(func, B0, z0, nb_splits, times, rhs, integrator, learning_rate, tol, maxiter):
# learning_rate=1., tol=1e-6, maxiter=3):
    grad = jax.jacfwd(func)

    B = B0

    ## Check if times contains numpy arrays and convert to floats
    if isinstance(times[0], np.float64):
        times = tuple(map(lambda x: float(x), times))

    # shape = B.shape
    # Nnz = shape[0]*shape[1]

    ## Print func name and its arguments
    # print("Function name: ", func.__name__, "\nLocal vars: ", func.__code__.co_varnames, "\nFreevars: ", func.__code__.co_freevars)

    # times = jnp.array(times)[:, jnp.newaxis]
    # func(B, z0, nb_splits, times, rhs, integrator)

    for k in range(maxiter):
        # grad_inv = jnp.linalg.inv(grad(B, z0).reshape((Nnz, Nnz)))
        # func_eval = func(B, z0).reshape((Nnz, 1))

        # print("Types of all arguments: ", type(rhs), type(B), type(z0), type(times), type(times[0]), type(integrator))

        print("PinT iteration counter: ", k)
        grad_inv = jnp.linalg.inv(grad(B, z0, nb_splits, times, rhs, integrator))
        func_eval = func(B, z0, nb_splits, times, rhs, integrator)

        B_new = B - learning_rate * grad_inv @ func_eval
        # B_new = B - learning_rate * func_eval

        # if jnp.linalg.norm(B_new - B) < tol:
        #     break
        B = B_new

    # print("Shape of the returned B is: ", B.shape)

    return B        ## TODO B is currentlyb of size N \times nz, but should only be of size nz



@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def direct_root_finder(func, B0, z0, nb_splits, times, rhs, integrator, learning_rate=1., tol=1e-6, maxiter=10):
    grad = jax.jacfwd(func)

    B = B0

    ## Print func name and its arguments
    # print("Function name: ", func.__name__, "\nLocal vars: ", func.__code__.co_varnames, "\nFreevars: ", func.__code__.co_freevars)

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
