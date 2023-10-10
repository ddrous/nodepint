#%%
## Shape agnostic ODE solvers for nodepint

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.ode import odeint

import numpy as np
import equinox as eqx

## The interface of each integrator is:
# - the function func,
# - the input vector, 
# - start and finish times
# - and the additional arguments to func.
# -* the custom jvp for the integrator: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html



## The default Jax differentiable integrator
# dopri_integrator = jax.jit(odeint, static_argnums=(0))
# dopri_integrator = odeint

def dopri_integrator(rhs_params, static, y0, t, hmax):      ## Inverts the order of t and y0 passed to func
    # print("Yo shape is: ", y0.shape)
    # print("rhs shapes:", rhs)

    # def rhs(y, t):
    #     return eqx.combine(rhs_params, static)(y, t)
    
    rhs = lambda y, t: eqx.combine(rhs_params, static)(y, t)

    return odeint(rhs, y0, t, rtol=1e-6, atol=1e-6, mxstep=1000, hmax=hmax)

# ## Simple Euler integrator
# def euler_step(rhs, y, t, dt):
#     ret = y+rhs(y, t)*dt, t+dt
#     return ret

# # @partial(jax.jit, static_argnums=(0, 2, 3))
# def euler_integrator(rhs_params, static, y0, t, hmax=1e-2):

#     rhs = eqx.combine(rhs_params, static)

#     # t = np.array(t)
#     # dt = jnp.min(jnp.minimum(jnp.ones_like(t[1:])*hmax, t[1:] - t[:-1]))
#     dt = np.min(np.minimum(np.ones_like(t[1:])*hmax, t[1:] - t[:-1]))
#     nb_iter = int((t[-1] - t[0]) / dt)

#     def body_func(i, yt):       ## TODO this is sooo not functional
#         newy, newt = euler_step(rhs, yt[i-1, 1:], yt[i-1, 0, jnp.newaxis], dt)
#         yt = yt.at[i, 0].set(newt[0])
#         yt = yt.at[i, 1:].set(newy)
#         return yt

#     yt = jnp.zeros((nb_iter, y0.shape[0]+1))
#     yt = yt.at[0, 0].set(t[0])
#     yt = yt.at[0, 1:].set(y0)

#     yt = jax.lax.fori_loop(1, nb_iter, body_func, yt)

#     return yt[:, 1:] 


## Inspired by http://implicit-layers-tutorial.org/implicit_functions/
def euler_integrator(rhs_params, static, y0, t, hmax):
  rhs = eqx.combine(rhs_params, static)
  def step(state, t):
    y_prev, t_prev = state
    dt = jnp.minimum(t - t_prev, hmax)
    y = y_prev + dt * rhs(y_prev, t_prev)
    return (y, t), y
  _, ys = lax.scan(step, (y0, t[0]), t[1:])
  return ys



def rk4_integrator(rhs_params, static, y0, t, hmax):
  rhs = eqx.combine(rhs_params, static)
  def step(state, t):
    y_prev, t_prev = state
    h = jnp.minimum(t - t_prev, hmax)
    k1 = h * rhs(y_prev, t_prev)
    k2 = h * rhs(y_prev + k1/2., t_prev + h/2.)
    k3 = h * rhs(y_prev + k2/2., t_prev + h/2.)
    k4 = h * rhs(y_prev + k3, t + h)
    y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y, t), y
  _, ys = lax.scan(step, (y0, t[0]), t[1:])
  return ys

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
