#%%
## Time-parallel strategies for nodepint

import jax
import jax.numpy as jnp


def define_mtp_shooting_func(nb_processors, times, integrator):
    ## Split times among N = nb_processors

    ## Construct solution vector B

    ## Define the actual shooting function
    def mtp_shooting_func(B, z0):
        pass

    return mtp_shooting_func


def select_root_finding_function(pint_scheme:str):

    if pint_scheme=="newton":
        shooting_func = newton_shooting       ## Use partial if necessary
    elif pint_scheme=="sequential":
        shooting_func = sequential_shooting

    return shooting_func



def solve_multiple_shooting(pint_scheme:str, B0, nb_processors, times, integrator):
    mtp_sht_func = define_mtp_shooting_func(nb_processors, times, integrator)
    shooting_func = select_root_finding_function(pint_scheme)

    # return shooting_func(mtp_sht_func, x0, method='hybr', jac=None, tol=None, callback=None, options=None)
    return shooting_func(mtp_sht_func, B0)


#%%

## Newton's method TODO define custom JVP rule https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
def newton_shooting(func, B0, learning_rate=1., tol=1e-6, maxiter=100):
    grad = jax.grad(func)

    B = B0
    for _ in range(maxiter):
        B_new = B - learning_rate * jnp.linalg.inv(grad(B)) @ func(B)
        if jnp.linalg.norm(B_new - B) < tol:
            break
        B = B_new

    return B



def sequential_shooting(func, B0):
    pass
