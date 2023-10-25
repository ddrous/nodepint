#%%
## Shape agnostic ODE solvers for nodepint

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import lax

import numpy as np
import equinox as eqx
from equinox.internal import while_loop


from functools import partial



import operator as op

from jax._src import core
from jax import custom_derivatives
from jax import lax
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.util import safe_map, safe_zip
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map
from jax._src import linear_util as lu

# import jax.

map = safe_map
zip = safe_zip




## The interface of each integrator is:
# - the function func,
# - the input vector, 
# - start and finish times
# - and the additional arguments to func.
# -* the custom jvp for the integrator: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html




# @partial(jax.jit, static_argnums=(0, 1))
def dopri_integrator(rhs_params, static, y0, t, rtol, atol, hmax, mxstep, max_steps_rev, kind):      ## Inverts the order of t and y0 passed to func

    rhs = lambda y, t: eqx.combine(rhs_params, static)(y, t)
    rhs = jax.jit(rhs)

    return odeint(rhs, y0, t, rtol=rtol, atol=atol, mxstep=mxstep, hmax=hmax)
    # return odeint(rhs, y0, t, hmax=hmax)




# @partial(jax.jit, static_argnums=(0, 1))
def euler_integrator(rhs_params, static, y0, t, rtol, atol, hmax, mxstep, max_steps_rev, kind):
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


# @partial(jax.jit, static_argnums=(0, 1))
def rk4_integrator(rhs_params, static, y0, t, rtol, atol, hmax, mxstep, max_steps_rev, kind):
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








# @partial(jax.jit, static_argnums=(0, 4,5,7,6))
# def dopri5(f, y0, t0, t_final, h0, hmax, atol=1e-2, rtol=1e-2):
#     def runge_kutta_step(y, t, h):
#         k1 = h * f(y, t)
#         k2 = h * f(y + k1/5, t + h/5)
#         k3 = h * f(y + 3*k1/40 + 9*k2/40, t + 3*h/10)
#         k4 = h * f(y + 44*k1/45 - 56*k2/15 + 32*k3/9, t + 4*h/5)
#         k5 = h * f(y + 19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729, t + 8*h/9)
#         k6 = h * f(y + 9017*k1/3168 - 355/33*k2 - 46732*k3/5247 + 49*k4/176 - 5103*k5/18656, t + h)
        
#         y_new = y + 35*k1/384 + 0*k2 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84
#         return y_new

#     def condition(carry):
#         _, t, _ = carry
#         return t < t_final

#     def body(carry):
#         y, t, h = carry
#         y1 = runge_kutta_step(y, t, h)
#         y2 = runge_kutta_step(y1, t + h, h)

#         error = jnp.max(jnp.abs(y2 - y1))
#         h_opt = h * jnp.sqrt(rtol / error)
#         h_new = jnp.minimum(jnp.minimum(h_opt, t_final - t), hmax)  # Consider the maximum time step

#         return (y1, t + h, h_new)

#     initial_state = (y0, t0, h0)
#     final_state = lax.while_loop(condition, body, initial_state)

#     return final_state[0], final_state[1]














def ravel_first_arg(f, unravel):
  return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped

@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
  y = unravel(y_flat)
  ans = yield (y,) + args, {}
  ans_flat, _ = ravel_pytree(ans)
  yield ans_flat

def interp_fit_dopri(y0, y1, k, dt):
  # Fit a polynomial to the results of a Runge-Kutta step.
  dps_c_mid = jnp.array([
      6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
      -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
      -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2], dtype=y0.dtype)
  y_mid = y0 + dt.astype(y0.dtype) * jnp.dot(dps_c_mid, k)
  return jnp.asarray(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))

def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
  dt = dt.astype(y0.dtype)
  a = -2.*dt*dy0 + 2.*dt*dy1 -  8.*y0 -  8.*y1 + 16.*y_mid
  b =  5.*dt*dy0 - 3.*dt*dy1 + 18.*y0 + 14.*y1 - 32.*y_mid
  c = -4.*dt*dy0 +    dt*dy1 - 11.*y0 -  5.*y1 + 16.*y_mid
  d = dt * dy0
  e = y0
  return a, b, c, d, e

def initial_step_size(fun, t0, y0, order, rtol, atol, f0):
  # Algorithm from:
  # E. Hairer, S. P. Norsett G. Wanner,
  # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
  y0, f0 = promote_dtypes_inexact(y0, f0)
  dtype = y0.dtype

  scale = atol + jnp.abs(y0) * rtol
  d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
  d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

  h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
  y1 = y0 + h0.astype(dtype) * f0
  f1 = fun(y1, t0 + h0)
  d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

  h1 = jnp.where((d1 <= 1e-15) & (d2 <= 1e-15),
                jnp.maximum(1e-6, h0 * 1e-3),
                (0.01 / jnp.maximum(d1, d2)) ** (1. / (order + 1.)))

  return jnp.minimum(100. * h0, h1)

def runge_kutta_step(func, y0, f0, t0, dt):
  # Dopri5 Butcher tableaux
  alpha = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0], dtype=dt.dtype)
  beta = jnp.array(
      [[1 / 5, 0, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
       [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
       [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
       [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
       [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]],
      dtype=f0.dtype)
  c_sol = jnp.array(
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
      dtype=f0.dtype)
  c_error = jnp.array([
      35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085, 125 / 192 -
      451 / 720, -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300, -1. / 60.
  ], dtype=f0.dtype)

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt.astype(f0.dtype) * jnp.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return k.at[i, :].set(ft)

  k = jnp.zeros((7, f0.shape[0]), f0.dtype).at[0, :].set(f0)
  k = lax.fori_loop(1, 7, body_fun, k)

  y1 = dt.astype(f0.dtype) * jnp.dot(c_sol, k) + y0
  y1_error = dt.astype(f0.dtype) * jnp.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def abs2(x):
  if jnp.iscomplexobj(x):
    return x.real ** 2 + x.imag ** 2
  else:
    return x ** 2

def mean_error_ratio(error_estimate, rtol, atol, y0, y1):
  err_tol = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
  err_ratio = error_estimate / err_tol.astype(error_estimate.dtype)
  return jnp.sqrt(jnp.mean(abs2(err_ratio)))

def optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0,
                      dfactor=0.2, order=5.0):
  """Compute optimal Runge-Kutta stepsize."""
  dfactor = jnp.where(mean_error_ratio < 1, 1.0, dfactor)

  factor = jnp.minimum(ifactor,
                      jnp.maximum(mean_error_ratio**(-1.0 / order) * safety, dfactor))
  return jnp.where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)




# def dopri_integrator_diff(params, static, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, hmax=jnp.inf):
# def dopri_integrator_diff(params, static, y0, t, *args, rtol=1.4e-1, atol=1.4e-1, hmax=jnp.inf, mxstep=10, max_steps_rev=2, kind="checkpointed"):
# def dopri_integrator_diff(params, static, y0, t, *args, rtol, atol, hmax, mxstep, max_steps_rev, kind):
def dopri_integrator_diff(params, static, y0, t, rtol, atol, hmax, mxstep, max_steps_rev, kind):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `jnp.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`, which must be arrays
      scalars, or (nested) standard Python containers (tuples, lists, dicts,
      namedtuples, i.e. pytrees) of those types.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).
    hmax: float, maximum step size allowed (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  # for arg in tree_leaves(args):
  #   if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
  #     raise TypeError(
  #       f"The contents of odeint *args must be arrays or scalars, but got {arg}.")
  if not jnp.issubdtype(t.dtype, jnp.floating):
    raise TypeError(f"t must be an array of floats, but got {t}.")

  # converted, consts = custom_derivatives.closure_convert(func, y0, t[0], *args)
  # return dopri5_wrapper(converted, rtol, atol, mxstep, hmax, y0, t, *args, *consts)


  func = eqx.combine(params, static)

  return dopri5_wrapper(func, rtol, atol, hmax, mxstep, max_steps_rev, kind, y0, t)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def dopri5_wrapper(func, rtol, atol, hmax, mxstep, max_steps_rev, kind, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  # func = eqx.combine(params, static)
  func = ravel_first_arg(func, unravel)
  # out = dopri5_core(params, static, rtol, atol, mxstep, hmax, y0, ts, *args)
  out = dopri5_core(func, rtol, atol, hmax, mxstep, max_steps_rev, kind, y0, ts, *args)
  return jax.vmap(unravel)(out)

# @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
# def dopri5_core(params, static, rtol, atol, mxstep, hmax, y0, ts, *args):
def dopri5_core(func, rtol, atol, hmax, mxstep, max_steps_rev, kind, y0, ts, *args):
  func_ = lambda y, t: func(y, t)
  # func_ = lambda y, t: eqx.combine(params, static)(y, t, *args)
  # func_ = lambda y, t: eqx.combine(params, static)(y, t)  ## !TODO WARNING! This doesnt use args arguments !!!!

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      next_y, next_f, next_y_error, k = runge_kutta_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratio = mean_error_ratio(next_y_error, rtol, atol, y, next_y)
      new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
      dt = jnp.clip(optimal_step_size(dt, error_ratio), a_min=0., a_max=hmax)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(jnp.where, error_ratio <= 1.), new, old)

    # _, *carry = lax.while_loop(cond_fun, body_fun, [0] + carry)
    # _, *carry = while_loop(cond_fun, body_fun, [0] + carry, max_steps=2, kind="bounded")
    _, *carry = while_loop(cond_fun, body_fun, [0] + carry, max_steps=max_steps_rev, kind=kind)

    _, _, t, _, last_t, interp_coeff = carry
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = jnp.polyval(interp_coeff, relative_output_time.astype(interp_coeff.dtype))
    return carry, y_target

  f0 = func_(y0, ts[0])
  dt = jnp.clip(initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0), a_min=0., a_max=hmax)
  interp_coeff = jnp.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff]
  _, ys = lax.scan(scan_fun, init_carry, ts[1:])
  return jnp.concatenate((y0[None], ys))























# %%

## %%timeit -n1 -r1 TODO Careful to only use this when running the script in iPython

if __name__ == "__main__":

    ## The Lorentz system
    # @jax.jit
    # def lorentz(u, t, sigma=10., beta=8./3, rho=28.):
    #     """Lorentz system"""
    #     x, y, z = u
    #     return jnp.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    

    class Lorentz(eqx.Module):
      sigma: float
      beta: float
      rho: float
      def __init__(self, sigma=10., beta=8./3, rho=28.):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
      def __call__(self, u, t, *args):
        """Lorentz system"""
        x, y, z = u
        # x, y, z = u[0], u[1], u[2]
        return jnp.array([self.sigma*(y-x), x*(self.rho-z)-y, x*y-self.beta*z])

    lorentz = Lorentz()
    params, static = eqx.partition(lorentz, eqx.is_array)

    t0, tf = 0., 100.   # Start and finish times
    times = jnp.linspace(t0, tf, 10000)     # Save locations for the solution
    hmax = 1e-2         # Maximum step size

    u0 = jnp.array([1., 1., 1.])    # Initial condition

    print("BENCHMARKS")
    # print("=== Jax's dopri odeint ===")
    # %timeit -n1 -r2 dopri_integrator(lorentz, u0, t=times[:])
    # print("=== Euler integration ===")
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

    ## Create a very large ax
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(20,20))



    print("=== Euler integration ===")

    us = euler_integrator(params, static, u0, t=times, hmax=hmax)
    # %time us = euler_integrator(params, static, u0, t=times, hmax=hmax)
    ax = sbplot(us[:,0], us[:,2], ".-", markersize=1, ax=ax, x_label="x", y_label="z", label="Euler Explicit", title="Lorentz's attractors", color="r", lw=1)



    print("=== Jax's dopri odeint ===")

    us = dopri_integrator(params, static, u0, t=times, hmax=hmax)
    # %time us = dopri_integrator(params, static, u0, t=times, hmax=hmax)
    ax = sbplot(us[:,0], us[:,2], label="Jax's RK", color="b", lw=2, ax=ax)


    print("=== Eqx's dopri odeint ===")
    # print(while_loop.__doc__)## Useful trick !!


    us = dopri_integrator_diff(params, static, u0, times, hmax=hmax)
    # %time us = dopri5_bounded(params, static, u0, times, hmax=hmax)
    ax = sbplot(us[:,0], us[:,2], "--", label="Eqx's RK", color="g", lw=5, ax=ax)

    ## Increase ax's legend size
    ax.legend(fontsize=30)




# %%
