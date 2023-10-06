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
    rhs = eqx.combine(rhs_params, static)

    ## Split times among N = nb_processors
    # times = np.array(times)[:, jnp.newaxis]
    # split_times = np.array_split(times, nb_splits)

    t0, tf, N = times
    N_ = N//nb_splits + 1

    Z_ = [z0]
    nz = z0.shape[0]

    for n in range(nb_splits):      ## TODO do this in parallel   
        t0_ = t0 + (n+0)*(tf-t0)/nb_splits
        tf_ = t0 + (n+1)*(tf-t0)/nb_splits
        t_ = np.linspace(t0_, tf_, N_)

        z_next = integrator(rhs, Z[n*nz:(n+1)*nz], t=t_)[-1,...]
        Z_.append(z_next)

    return Z - jnp.concatenate(Z_, axis=0)
    # return -Z + jnp.concatenate(Z_, axis=0)
    # return jnp.concatenate(Z_, axis=0)          ## TODO remember this is a shooting function, so look above !



#%%

## Newton's method - TODO we still need to define:
# - custom JVP rule https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
# - example of fixed-point interation in PyTorch http://implicit-layers-tutorial.org/deep_equilibrium_models/
# @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))    ## TODO remember to jit
def newton_root_finder(func, B0, z0, nb_splits, times, rhs, integrator, learning_rate, tol, max_iter):
    grad = jax.jacfwd(func)

    B = B0

    for k in range(max_iter):       ## MAX_ITER is N (see Massaroli)

        ## This also tells us whether we are recompiling
        print("PinT iteration counter: ", k)

        grad_inv = jnp.linalg.inv(grad(B, z0, nb_splits, times, rhs, integrator))
        func_eval = func(B, z0, nb_splits, times, rhs, integrator)

        B_new = B - learning_rate * grad_inv @ func_eval

        # if jnp.linalg.norm(B_new - B) < tol:  ## TODO not jit-friendly
        #     break

        B = B_new

    return B


# @jax.custom_vjp
# @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))    ## TODO remember to jit
# @partial(jax.custom_vjp, nondiff_argnums=(0,3,4,6,7,8,9))
def direct_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):       ## See Massaroli
    grad = jax.jacfwd(func)


    ## Pirnt the type of every single argument of this function
    B = B0

    for k in range(max_iter):
        ## Solve a linear system

        print("PinT iteration counter: ", k)

        grad_eval = grad(B, z0, nb_splits, times, rhs, integrator)
        func_eval = func(B, z0, nb_splits, times, rhs, integrator)

        B_new = B + jnp.linalg.solve(grad_eval, -func_eval)

        # if jnp.linalg.norm(B_new - B) < tol:
        #     break

        B = B_new

    return B



    # return fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)




def direct_fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):       ## See Massaroli
    ## Wrapper to use a shooting function as input like other APIs

    fp_func = lambda B, z0, nb_splits, times, rhs, static, integrator: -B + func(B, z0, nb_splits, times, rhs, static, integrator)

    return fixed_point_finder(fp_func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)




def sequential_root_finder(func, z0, B0):
    pass











# @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
# @partial(jax.custom_vjp, nondiff_argnums=(0,2,3,4,6,7,8,9))
@partial(jax.custom_vjp, nondiff_argnums=(0,3,4,6,7,8,9,10))
def fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    # jax.debug.breakpoint()

    def cond_fun(carry):
        B_prev, B = carry
        return jnp.linalg.norm(B_prev - B) > 1e-2

    def body_fun(carry):
        _, B = carry
        return B, func(B, z0, nb_splits, times, rhs, static,integrator)

    _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, func(B0, z0, nb_splits, times, rhs, static, integrator)))
    return B_star


# @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def fixed_point_finder_fwd(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    # B_star = fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static,integrator, learning_rate, tol, max_iter)

    # zero_func = lambda B, z0, nb_splits, times, rhs, integrator: B-func(B, z0, nb_splits, times, rhs, integrator)
    # B_star = direct_root_finder(zero_func, B0, z0, nb_splits, times, rhs, static,integrator, learning_rate, tol, max_iter)
    B_star = fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)

    # return B_star, (B_star, z0, nb_splits, times, rhs, integrator, learning_rate, tol, max_iter)
    return B_star, (B_star, z0, rhs)




# def rev_iter(func, packed, w):
#     z0, nb_splits, times, rhs, integrator, B_star, v = packed
#     _, vjp_B = jax.vjp(lambda B: func(B, z0, nb_splits, times, rhs, integrator), B_star)
#     return v + vjp_B(w)[0]


def inner_fixed_point_finder(func, B_star, z0, nb_splits, times, rhs, static, integrator, v, learning_rate, tol, max_iter):

    _, vjp_B = jax.vjp(lambda B: func(B, z0, nb_splits, times, rhs, static, integrator), B_star)

    def cond_fun(carry):
        w_prev, w = carry
        return jnp.linalg.norm(w_prev - w) > 1e-2

    def body_fun(carry):
        _, w = carry
        return w, v + vjp_B(w)[0]

    _, w_star = jax.lax.while_loop(cond_fun, body_fun, (v, v + vjp_B(v)[0]))
    return w_star



def fixed_point_finder_bwd(func, nb_splits, times, static, integrator, learning_rate, tol, max_iter, res, v):
    B_star, z0, rhs = res

# def fixed_point_finder_bwd(func, res, v):
#     B_star, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter = res

    _, vjp_theta = jax.vjp(lambda theta: func(B_star, z0, nb_splits, times, theta, static, integrator), rhs)

    # w = fixed_point_finder(partial(rev_iter, func),
    #                         (z0, nb_splits, times, rhs, static, integrator, B_star, v),
    #                          v)

    w = inner_fixed_point_finder(func, B_star, z0, nb_splits, times, rhs, static, integrator, v, learning_rate, tol, max_iter)

    theta_bar, = vjp_theta(w)

    return jnp.zeros_like(B_star), None, theta_bar


fixed_point_finder.defvjp(fixed_point_finder_fwd, fixed_point_finder_bwd)
