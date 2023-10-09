#%%
## Time-parallel strategies for nodepint

import jax
import jax.numpy as jnp
from functools import partial
import equinox as eqx

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


def shooting_function(Z, z0, nb_splits, times, rhs_params, static, integrator):

    ## Rebuild the equinox model
    # rhs = eqx.combine(rhs_params, static)

    t0, tf, N = times
    N_ = N//nb_splits + 1

    Z_ = [z0]
    nz = z0.shape[0]

    for n in range(nb_splits):      ## TODO do this in parallel   
        t0_ = t0 + (n+0)*(tf-t0)/nb_splits
        tf_ = t0 + (n+1)*(tf-t0)/nb_splits
        t_ = np.linspace(t0_, tf_, N_)

        # z_next = integrator(rhs, Z[n*nz:(n+1)*nz], t=t_)[-1,...]
        z_next = integrator(rhs_params, static, Z[n*nz:(n+1)*nz], t=t_)[-1,...]
        Z_.append(z_next)

    return Z - jnp.concatenate(Z_, axis=0)


#%%

def newton_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    grad = jax.jacfwd(func)

    # B = B0
    # for k in range(max_iter):       ## MAX_ITER is N (see Massaroli)

    #     ## This also tells us whether we are recompiling
    #     print("PinT iteration counter: ", k)

    #     grad_inv = jnp.linalg.inv(grad(B, z0, nb_splits, times, rhs, integrator))
    #     func_eval = func(B, z0, nb_splits, times, rhs, integrator)

    #     B_new = B - learning_rate * grad_inv @ func_eval

    #     # if jnp.linalg.norm(B_new - B) < tol:  ## TODO not jit-friendly
    #     #     break

    #     B = B_new
    # B_star = B

    def cond_fun(carry):
        B_prev, B = carry
        return jnp.linalg.norm(B_prev - B) > 1e-4

    def body_fun(carry):
        _, B = carry
        grad_inv = jnp.linalg.inv(grad(B, z0, nb_splits, times, rhs, static, integrator))
        func_eval = func(B, z0, nb_splits, times, rhs, static, integrator)
        return B, B - learning_rate * grad_inv @ func_eval

    _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, func(B0, z0, nb_splits, times, rhs, static, integrator)))

    return B_star



def direct_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    grad = jax.jacfwd(func)

    def cond_fun(carry):
        B_prev, B = carry
        return jnp.linalg.norm(B_prev - B) > 1e-4       ## TODO Add a tolerance or a maxiter

    def body_fun(carry):
        _, B = carry
        grad_eval = grad(B, z0, nb_splits, times, rhs, static, integrator)
        func_eval = func(B, z0, nb_splits, times, rhs, static, integrator)
        return B, B + jnp.linalg.solve(grad_eval, -func_eval)

    _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, func(B0, z0, nb_splits, times, rhs, static, integrator)))


    return B_star


def fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):  ## !TODO name this as fixed point AD

    fp_func = lambda B, z0, nb_splits, times, rhs, static, integrator: -B + func(B, z0, nb_splits, times, rhs, static, integrator)

    def cond_fun(carry):
        B_prev, B = carry
        return jnp.linalg.norm(B_prev - B) > 1e-2

    def body_fun(carry):
        _, B = carry
        return B, fp_func(B, z0, nb_splits, times, rhs, static, integrator)

    _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, fp_func(B0, z0, nb_splits, times, rhs, static, integrator)))
    return B_star



def sequential_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    pass


def parareal(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    pass





@partial(jax.custom_vjp, nondiff_argnums=(0,3,4,6,7,8,9,10,11))
def fixed_point_ad(func, B0, z0, nb_splits, times, rhs, static, integrator, pint_scheme, learning_rate, tol, max_iter):  ## !TODO name this as fixed point AD

    # fp_func = lambda B, z0, nb_splits, times, rhs, static, integrator: -B + func(B, z0, nb_splits, times, rhs, static, integrator)

    # def cond_fun(carry):
    #     B_prev, B = carry
    #     return jnp.linalg.norm(B_prev - B) > 1e-2

    # def body_fun(carry):
    #     _, B = carry
    #     return B, fp_func(B, z0, nb_splits, times, rhs, static, integrator)

    # _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, fp_func(B0, z0, nb_splits, times, rhs, static, integrator)))

    B_star = pint_scheme(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)

    return B_star


def fixed_point_ad_fwd(func, B0, z0, nb_splits, times, rhs, static, integrator, pint_scheme, learning_rate, tol, max_iter):
    ## TODO should I add an argument "pint_scheme" for the fixed point/root function ?

    # B_star = fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)
    # B_star = direct_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)
    # B_star = newton_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)

    B_star = pint_scheme(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)

    return B_star, (B_star, z0, rhs)


def inner_fixed_point_ad(func, B_star, z0, nb_splits, times, rhs, static, integrator, v, learning_rate, tol, max_iter):

    _, vjp_B = jax.vjp(lambda B: func(B, z0, nb_splits, times, rhs, static, integrator), B_star)

    def cond_fun(carry):
        w_prev, w = carry
        return jnp.linalg.norm(w_prev - w) > 1e-2

    def body_fun(carry):
        _, w = carry
        return w, v + vjp_B(w)[0]

    _, w_star = jax.lax.while_loop(cond_fun, body_fun, (v, v + vjp_B(v)[0]))
    return w_star



def fixed_point_ad_bwd(func, nb_splits, times, static, integrator, pint_scheme, learning_rate, tol, max_iter, res, v):
    B_star, z0, rhs = res

    fp_func = lambda B, z0, nb_splits, times, rhs, static, integrator: -B + func(B, z0, nb_splits, times, rhs, static, integrator)

    _, vjp_theta = jax.vjp(lambda theta: fp_func(B_star, z0, nb_splits, times, theta, static, integrator), rhs)

    w = inner_fixed_point_ad(fp_func, B_star, z0, nb_splits, times, rhs, static, integrator, v, learning_rate, tol, max_iter)

    theta_bar, = vjp_theta(w)

    return jnp.zeros_like(B_star), None, theta_bar


fixed_point_ad.defvjp(fixed_point_ad_fwd, fixed_point_ad_bwd)
