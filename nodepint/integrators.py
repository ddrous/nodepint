#%%
## Shape agnostic ODE solvers for nodepint

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import numpy as np
import equinox as eqx

from functools import partial

## The interface of each integrator is:
# - the function func,
# - the input vector, 
# - start and finish times
# - and the additional arguments to func.
# -* the custom jvp for the integrator: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html



## The default Jax differentiable integrator


# @partial(jax.jit, static_argnums=(0, 1))
def dopri_integrator(rhs_params, static, y0, t, hmax):      ## Inverts the order of t and y0 passed to func

    rhs = lambda y, t: eqx.combine(rhs_params, static)(y, t)
    rhs = jax.jit(rhs)

    return odeint(rhs, y0, t, rtol=1e-4, atol=1e-4, mxstep=100, hmax=hmax)


# @partial(jax.jit, static_argnums=(0, 1))
def euler_integrator(rhs_params, static, y0, t, hmax):
  """hmax is never used, but is here for compatibility with other integrators """
  rhs = eqx.combine(rhs_params, static)
  def step(state, t):
    y_prev, t_prev = state
    dt = t - t_prev
    y = y_prev + dt * rhs(y_prev, t_prev)
    return (y, t), y
  _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
  # return ys
  return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)


@partial(jax.jit, static_argnums=(0, 1))
def rk4_integrator(rhs_params, static, y0, t, hmax):
  rhs = eqx.combine(rhs_params, static)
  def step(state, t):
    y_prev, t_prev = state
    h = t - t_prev
    k1 = h * rhs(y_prev, t_prev)
    k2 = h * rhs(y_prev + k1/2., t_prev + h/2.)
    k3 = h * rhs(y_prev + k2/2., t_prev + h/2.)
    k4 = h * rhs(y_prev + k3, t + h)
    y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y, t), y
  _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
  # return ys
  return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)


## RBF integrator (TODO Implement this from Updec)



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
    # %timeit -n1 -r2 dopri_integrator(lorentz, u0, t=times[:])
    print("=== Euler integration ===")
    # %timeit -n1 -r2 euler_integrator(lorentz, u0, t=times[:], hmax=hmax)

    ## Plot the attractors with pyvista
    # from utils import pvplot

    # us = dopri_integrator(lorentz, u0, t=times)
    # ax = pvplot(us[:,0], us[:,2], label="Jax's RK", show=False, color="b", width=2, style="-")

    # # u0 = jnp.array([1., 1.001, 1.])
    # us = euler_integrator(lorentz, u0, t=times, hmax=hmax)
    # ax = pvplot(us[:,0], us[:,2], ax=ax, xlabel="x", ylabel="z", label="Euler Explicit", title="Lorentz's attractors", color="r", width=1, style="-")

    # Plot the attractors with seaborn
    from utils import sbplot

    us = dopri_integrator(lorentz, u0, t=times, hmax=hmax)
    ax = sbplot(us[:,0], us[:,2], label="Jax's RK", color="b", lw=2)

    us = euler_integrator(lorentz, u0, t=times, hmax=hmax)
    ax = sbplot(us[:,0], us[:,2], ".-", markersize=2, ax=ax, x_label="x", y_label="z", label="Euler Explicit", title="Lorentz's attractors", color="r", lw=1)


# %%
