#%%
## Time-parallel strategies for nodepint

import jax
import jax.numpy as jnp


def define_shooting_function(nb_processors, times, rhs, integrator):
    ## Split times among N = nb_processors
    splittimes = jnp.array_split(times, nb_processors)

    ## Define the actual shooting function
    def mtp_shooting_func(Z, z0):

        Z_ = [z0]

        for n in range(nb_processors):
            ts = splittimes[n]
            if n < nb_processors-1:
                ts = jnp.concatenate([ts, [splittimes[n+1]]])

            znext = integrator(rhs, Z[n], times=ts)
            Z_.append(znext)

        return Z - jnp.array(Z_)

    return mtp_shooting_func


def select_root_finding_function(pint_scheme:str):

    if pint_scheme=="newton":
        shooting_func = newton_shooting       ## Use partial if necessary
    elif pint_scheme=="sequential":
        shooting_func = sequential_shooting
    else:
        raise ValueError("Unknown time-parallel scheme")

    return shooting_func



def solve_multiple_shooting(pint_scheme:str, B0, nb_processors, times, integrator):
    mtp_sht_func = define_shooting_function(nb_processors, times, integrator)
    shooting_func = select_root_finding_function(pint_scheme)

    # return shooting_func(mtp_sht_func, x0, method='hybr', jac=None, tol=None, callback=None, options=None)
    return shooting_func(mtp_sht_func, B0)


#%%

## Newton's method - TODO we still need to define:
# - custom JVP rule https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
# - example of fixed-point interation in PyTorch http://implicit-layers-tutorial.org/deep_equilibrium_models/
def newton_shooting(func, B0, learning_rate=1., tol=1e-6, maxiter=10):
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
