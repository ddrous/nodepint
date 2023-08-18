#%%
## Shape agnostic ODE solvers for nodepint

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from functools import partial

## The interface of each integrator is:
# - the function func,
# - the input vector, 
# - start and finish times
# - and the additional arguments to func.
# -* the custom jvp for the integrator: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html



## The fault Jax differentiable integrator (TODO Jit this)
# dopri_integrator = jax.jit(odeint, static_argnums=(0))
dopri_integrator = odeint       ## TODO hmax only available with recent versions of JAX

## Simple Euler integrator (TODO Use ForI or LaxScan to make it faster)
@partial(jax.jit, static_argnums=(0))
def euler_step(func, y, t, dt):
    return y+func(y, t)*dt, t + dt

def euler_integrator(func, y0, t, hmax=1e-2):

    ys = []
    y = y0
    curr_t = t[0]
    dt = min((hmax, jnp.min(t[1:] - t[:-1])))

    while curr_t < t[-1]:

        y, curr_t = euler_step(func, y, curr_t, dt)
        ys.append(y)

    return jnp.stack(ys, axis=0)


## RBF integrator (TODO from Updec)



# %%
## %%timeit -n1 -r1 TODO Careful to only use this when running the script in iPython

if __name__ == "__main__":

    ## The Lorentz system
    @jax.jit
    def lorentz(u, t, sigma=10., beta=8./3, rho=28.):
        """Lorentz system"""
        x, y, z = u
        return jnp.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    
    t0, tf = 0., 100.   # Start and finish times
    times = jnp.linspace(t0, tf, 10000)     # Save locations for the solution
    hmax = 1e-1         # Maximum step size

    u0 = jnp.array([1., 1., 1.])    # Initial condition

    print("BENCHMARKS")
    print("=== Jax's dopri odeint ===")
    %timeit -n1 -r2 dopri_integrator(lorentz, u0, t=times[:])
    print("=== Euler integration ===")
    %timeit -n1 -r2 euler_integrator(lorentz, u0, t=times[:], hmax=hmax)

    ## Plot the attractors with pyvista
    from utils import pvplot

    us = dopri_integrator(lorentz, u0, t=times)
    ax = pvplot(us[:,0], us[:,2], label="Jax's RK", show=False, color="b", width=2, style="-")

    # u0 = jnp.array([1., 1.001, 1.])
    us = euler_integrator(lorentz, u0, t=times, hmax=hmax)
    ax = pvplot(us[:,0], us[:,2], ax=ax, xlabel="x", ylabel="z", label="Euler Explicit", title="Lorentz's attractors", color="r", width=1, style="-")

    # # Plot the attractors with seaborn
    # from utils import sbplot

    # us = dopri_integrator(lorentz, u0, t=times, hmax=hmax)
    # ax = sbplot(us[:,0], us[:,2], label="Jax's RK", color="b", lw=2)

    # us = euler_integrator(lorentz, u0, t=times, hmax=hmax)
    # ax = sbplot(us[:,0], us[:,2], ".-", markersize=2, ax=ax, x_label="x", y_label="z", label="Euler Explicit", title="Lorentz's attractors", color="r", lw=1)


# %%
