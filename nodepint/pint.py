#%%
## Time-parallel strategies for nodepint

import jax
import jax.numpy as jnp



def solve_multiple_shooting(pint_scheme:str, B0, nb_processors, times, integrator):
    mtp_sht_func = define_shooting_function(nb_processors, times, integrator)
    shooting_func = select_root_finding_function(pint_scheme)

    # return shooting_func(mtp_sht_func, x0, method='hybr', jac=None, tol=None, callback=None, options=None)
    return shooting_func(mtp_sht_func, B0)



def select_root_finding_function(pint_scheme:str):

    if pint_scheme=="newton":
        shooting_func = newton_shooting       ## Use partial if necessary
    elif pint_scheme=="sequential":
        shooting_func = sequential_shooting
    else:
        raise ValueError("Unknown time-parallel scheme")

    return shooting_func



def define_shooting_function(nb_splits, times, rhs, integrator):
    ## Split times among N = nb_processors
    split_times = jnp.array_split(times, nb_splits)


    ## Define the actual shooting function
    def mtp_shooting_func(Z, z0):

        # print("mm")
        # print("All Arguments: ", Z, z0)

        Z_ = [z0]
        nz = z0.shape[0]

        for n in range(nb_splits):      ## TODO do this in parallel   
            ts = split_times[n]
            if n < nb_splits-1:
                ts = jnp.concatenate([ts, split_times[n+1][0, jnp.newaxis]])      ## TODO do this just once, and reuse the ts later on

            # z_next = integrator(rhs, Z[n], t=ts)[-1,...]
            z_next = integrator(rhs, Z[n*nz:(n+1)*nz], t=ts)[-1,...]
            Z_.append(z_next)

        # return Z - jnp.stack(Z_, axis=0)
        return Z - jnp.concatenate(Z_, axis=0)

    return mtp_shooting_func


#%%

## Newton's method - TODO we still need to define:
# - custom JVP rule https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
# - example of fixed-point interation in PyTorch http://implicit-layers-tutorial.org/deep_equilibrium_models/
def newton_shooting(func, z0, B0=None, learning_rate=1., tol=1e-6, maxiter=10):
    grad = jax.jacfwd(func)

    B = B0

    # shape = B.shape
    # Nnz = shape[0]*shape[1]

    ## Print func name and its arguments
    # print("Function name: ", func.__name__, "\nLocal vars: ", func.__code__.co_varnames, "\nFreevars: ", func.__code__.co_freevars)

    for _ in range(maxiter):
        # grad_inv = jnp.linalg.inv(grad(B, z0).reshape((Nnz, Nnz)))
        # func_eval = func(B, z0).reshape((Nnz, 1))

        grad_inv = jnp.linalg.inv(grad(B, z0))
        func_eval = func(B, z0)

        B_new = B - learning_rate * grad_inv @ func_eval
        if jnp.linalg.norm(B_new - B) < tol:
            break
        B = B_new

    # print("Shape of the returned B is: ", B.shape)

    return B        ## TODO B is currentlyb of size N \times nz, but should only be of size nz



def direct_shooting(func, z0, B0=None, learning_rate=1., tol=1e-6, maxiter=10):
    grad = jax.jacfwd(func)

    B = B0

    ## Print func name and its arguments
    # print("Function name: ", func.__name__, "\nLocal vars: ", func.__code__.co_varnames, "\nFreevars: ", func.__code__.co_freevars)

    for _ in range(maxiter):
        ## Solve a linear system
        B_new = jnp.linalg.solve(  grad(B, z0), B - func(B, z0))        ## FIx this


        if jnp.linalg.norm(B_new - B) < tol:
            break
        B = B_new

    return B



def sequential_shooting(func, z0, B0):
    pass
