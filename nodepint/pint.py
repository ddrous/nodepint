#%%
## Time-parallel strategies for nodepint

import jax
import jax.numpy as jnp
import equinox as eqx

import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding

from functools import partial
import numpy as np

from .utils import increase_timespan
from .integrators import euler_integrator, dopri_integrator, rk4_integrator

def select_root_finding_function(pint_scheme:str):

    if pint_scheme=="newton":
        root_finder = newton_root_finder
    elif pint_scheme=="direct":
        root_finder = direct_root_finder       ## Use partial if necessary
    elif pint_scheme=="direct_aug":
        root_finder = direct_root_finder_aug
    else:
        raise ValueError("Unknown time-parallel scheme")

    return root_finder


def shooting_function_serial(Z, z0, nb_splits, times, rhs_params, static, integrator):

    ## Rebuild the equinox model
    # rhs = eqx.combine(rhs_params, static)

    t0, tf, N = times[:3]
    hmax = times[3] if len(times)>3 else 1e-2
    N_ = N//nb_splits + 1

    Z_ = [z0]
    nz = z0.shape[0]

    ## TODO (Maybe) parallel disguised as serial
    for n in range(nb_splits):      ## TODO do this in parallel   
        t0_ = t0 + (n+0)*(tf-t0)/nb_splits
        tf_ = t0 + (n+1)*(tf-t0)/nb_splits
        t_ = np.linspace(t0_, tf_, N_)

        # z_next = integrator(rhs, Z[n*nz:(n+1)*nz], t=t_)[-1,...]
        z_next = integrator(rhs_params, static, Z[n*nz:(n+1)*nz], t_, hmax)[-1,...]
        Z_.append(z_next)
    Z_ = jnp.concatenate(Z_, axis=0)

    ## Actual parallel stuff
    # t = np.linspace(t0, tf, N)
    # Z_ = integrator(rhs_params, static, z0[None, ...], t, hmax)[-1,...]

    return Z - Z_



# def shooting_function_parallel(Z, z0, nb_splits, times, rhs_params, static, integrator):

#     t0, tf, N = times[:3]
#     hmax = times[3] if len(times)>3 else 1e-2
#     N_ = N//nb_splits + 1

#     # nz = z0.shape[0]
#     # t_s = []
#     # z0_s = []
#     # for n in range(nb_splits):
#     #     t0_ = t0 + (n+0)*(tf-t0)/nb_splits
#     #     tf_ = t0 + (n+1)*(tf-t0)/nb_splits
#     #     t_ = np.linspace(t0_, tf_, N_)

#     #     t_s.append(t_)
#     #     z0_s.append(Z[n*nz:(n+1)*nz])
    
#     nb_devices = jax.local_device_count()
#     devices = mesh_utils.create_device_mesh((nb_devices, 1))
#     shard = sharding.PositionalSharding(devices)

#     n_s = jnp.arange(nb_splits)
#     t0_s = t0 + n_s*(tf-t0)/nb_splits
#     tf_s = t0 + (n_s+1)*(tf-t0)/nb_splits
#     t_s = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, N_), in_axes=(0, 0))(t0_s, tf_s)

#     z0_s, t_s = jax.device_put((Z[:-1, :], t_s), shard)
#     Z_next = jax.vmap(integrator, in_axes=(None, None, 0, 0, None))(rhs_params, static, z0_s, t_s, hmax)

#     ## Only keep the last value per device, plus z0
#     Z_next = jnp.concatenate([z0[None, ...], Z_next[:, -1, ...]], axis=0)

#     return Z - Z_next





def shooting_function_parallel(Z, z0, nb_splits, times, rhs_params, static, integrator):

    t0, tf, N = times[:3]
    hmax = times[3] if len(times)>3 else 1e-2
    N_ = N//nb_splits + 1

    # nz = z0.shape[0]
    # t_s = []
    # z0_s = []
    # for n in range(nb_splits):
    #     t0_ = t0 + (n+0)*(tf-t0)/nb_splits
    #     tf_ = t0 + (n+1)*(tf-t0)/nb_splits
    #     t_ = np.linspace(t0_, tf_, N_)

    #     t_s.append(t_)
    #     z0_s.append(Z[n*nz:(n+1)*nz])
    
    nb_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((nb_devices, 1))
    shard = sharding.PositionalSharding(devices)

    n_s = jnp.arange(nb_splits)
    t0_s = t0 + n_s*(tf-t0)/nb_splits
    tf_s = t0 + (n_s+1)*(tf-t0)/nb_splits
    t_s = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, 2), in_axes=(0, 0))(t0_s, tf_s)

    z0_s, t_s = jax.device_put((Z[:-1, :], t_s), shard)
    Z_next = jax.vmap(integrator, in_axes=(None, None, 0, 0, None, None, None, None, None, None))(rhs_params, static, z0_s, t_s, 1e-1, 1e-1, jnp.inf, 20, 2, "checkpointed")

    ## Only keep the last value per device, plus z0
    Z_next = jnp.concatenate([z0[None, ...], Z_next[:, -1, ...]], axis=0)

    return Z - Z_next




# ## TODO struglles to duplicate the static arrays. Cannot convert list of funcs to arrays
# def shooting_function_pmap(Z, z0, nb_splits, times, rhs_params, static, integrator):

#     ## Rebuild the equinox model
#     # rhs = eqx.combine(rhs_params, static)

#     t0, tf, N = times
#     N_ = N//nb_splits + 1

#     Z_ = [z0]
#     nz = z0.shape[0]

#     t_s = []
#     z0_s = []
#     for n in range(nb_splits):      ## TODO do this in parallel   
#         t0_ = t0 + (n+0)*(tf-t0)/nb_splits
#         tf_ = t0 + (n+1)*(tf-t0)/nb_splits
#         t_ = np.linspace(t0_, tf_, N_)

#         t_s.append(t_)
#         z0_s.append(Z[n*nz:(n+1)*nz])
    
#     t_s = jnp.stack(t_s, axis=0)
#     z0_s = jnp.stack(z0_s, axis=0)

#     # print("Rhs is :", rhs_params)
#     # print("Static is :", static)

#     # p_rhs_params = jax.tree_map(lambda x: [x] * nb_splits, rhs_params)
#     p_rhs_params = jax.tree_map(lambda x: jnp.array([x] * nb_splits), rhs_params)
#     # p_static = jax.tree_map(lambda x: jnp.array([x] * nb_splits), static)
#     p_static = jax.tree_map(lambda x: [x] * nb_splits, static)

#     ## Parallelise accross devices
#     # Z_next = jax.pmap(integrator, static_broadcasted_argnums=2)(p_rhs_params, p_static, z0_s, t=t_s)
#     Z_next = jax.pmap(integrator, in_axes=(0, None, 0, 0, None))(p_rhs_params, static, z0_s, t_s, 1e-2)

#     ## Only keep the last value per device, plus z0
#     Z_next = jnp.concatenate([z0[None, ...], Z_next[:, -1, ...]], axis=0)

#     return Z - Z_next



#%%



def fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):  ## !TODO name this as fixed point AD

    fp_func = lambda B, z0, nb_splits, times, rhs, static, integrator: -B + func(B, z0, nb_splits, times, rhs, static, integrator)

    def cond_fun(carry):
        B_prev, B = carry
        return jnp.linalg.norm(B_prev - B) > tol    ## TODO: Loops bounds https://github.com/google/jax/pull/13062 

    def body_fun(carry):
        _, B = carry
        return B, fp_func(B, z0, nb_splits, times, rhs, static, integrator)

    _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, fp_func(B0, z0, nb_splits, times, rhs, static, integrator)))
    return B_star



def newton_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    # grad = jax.jacfwd(func)   ## TODO jax has a pb mixing fwd and rev
    grad = jax.jacrev(func)

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
        return jnp.linalg.norm(B_prev - B) > tol

    def body_fun(carry):
        _, B = carry
        grad_inv = jnp.linalg.inv(grad(B, z0, nb_splits, times, rhs, static, integrator))
        func_eval = func(B, z0, nb_splits, times, rhs, static, integrator)
        return B, B - learning_rate * grad_inv @ func_eval

    _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, func(B0, z0, nb_splits, times, rhs, static, integrator)))

    return B_star



def direct_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter):
    # grad = jax.jacfwd(func)   ## TODO jax has a pb mixing fwd and rev
    grad = jax.jacrev(func)

    def cond_fun(carry):
        B_prev, B = carry
        return jnp.linalg.norm(B_prev - B) > tol       ## TODO Add a tolerance or a maxiter

    def body_fun(carry):
        _, B = carry
        grad_eval = grad(B, z0, nb_splits, times, rhs, static, integrator)
        func_eval = func(B, z0, nb_splits, times, rhs, static, integrator)
        return B, B + jnp.linalg.solve(grad_eval, -func_eval)

    _, B_star = jax.lax.while_loop(cond_fun, body_fun, (B0, func(B0, z0, nb_splits, times, rhs, static, integrator)))


    return B_star



###### TODO This version only works for 1D problems
# def direct_root_finder_aug(func, B0, z0, nb_splits, times, rhs_params, static, integrator, learning_rate, tol, max_iter):
#     """ Direct root finder augmented with forward sensitivity analysis. Much more streamlined. The "func" is not used anymore. """

#     nz = z0.shape[0]
#     rhs = eqx.combine(rhs_params, static)
#     grad_U = jax.jacrev(rhs, argnums=0)

#     def aug_rhs(y, t):
#         U = y[:nz]
#         V = y[nz:].reshape((nz, nz))
#         V_bar = grad_U(U, t) @ V
#         return jnp.concatenate([rhs(U, t), V_bar.flatten()], axis=0)

#     t0, tf, N = times[:3]
#     hmax = times[3] if len(times)>3 else 1e-2
#     N_ = N//nb_splits + 1

#     nb_devices = jax.local_device_count()
#     devices = mesh_utils.create_device_mesh((nb_devices, 1))
#     shard = sharding.PositionalSharding(devices)

#     n_s = np.arange(nb_splits)
#     t0_s = t0 + n_s*(tf-t0)/nb_splits
#     tf_s = t0 + (n_s+1)*(tf-t0)/nb_splits
#     t_s = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, N_), in_axes=(0, 0))(t0_s, tf_s)
#     t_s = jax.device_put(t_s, shard)

#     def cond_fun(carry):
#         _, _, errors, k = carry
#         return (errors[k] > tol) & (k<max_iter)

#     def body_fun(carry):
#         _, U, errors, k = carry
  
#         V = jnp.broadcast_to(jnp.diag(jnp.ones(nz)).flatten(), (nb_splits, nz*nz))
#         UV0_s = jnp.concatenate([U[:-1,:], V], axis=1)
#         UV0_s = jax.device_put((UV0_s), shard)

#         UVf = jax.vmap(integrator, in_axes=(None, None, 0, 0, None))(aug_rhs, static, UV0_s, t_s, hmax)[:, -1, ...]

#         def step(U_kp1_n, n):
#             V_prev = UVf[n-1, nz:].reshape((nz, nz))
#             U_kp1_np1 = UVf[n-1, :nz] + V_prev@(U_kp1_n - U[n-1, :])
#             return (U_kp1_np1), U_kp1_np1

#         _, U_next = jax.lax.scan(step, (z0), n_s+1)

#         U_next = jnp.concatenate([z0[None, :], U_next[:, :]], axis=0)
#         errors = errors.at[k+1].set(jnp.linalg.norm(U_next - U))

#         return U, U_next, errors, k+1

#     errors = jnp.inf * jnp.ones((max_iter+1,))
#     errors = errors.at[0].set(2*tol)

#     _, U_star, errors, nb_iters = jax.lax.while_loop(cond_fun, body_fun, (B0, B0, errors, 0))

#     return U_star, errors[1:], nb_iters





def direct_root_finder_aug(func, B0, z0, nb_splits, times, rhs_params, static, integrator, learning_rate, tol, max_iter):
    """ Direct root finder augmented with forward sensitivity analysis. Much more streamlined. The "func" is not used anymore. """

    orig_shape = z0.shape
    z0 = z0.flatten()

    nz = jnp.size(z0)
    rhs = lambda y, t: eqx.combine(rhs_params, static)(jnp.reshape(y, orig_shape), t).flatten()
    grad_U = jax.jacrev(rhs, argnums=0)

    def aug_rhs(y, t):
        U = y[:nz]
        V = y[nz:].reshape((nz, nz))
        V_bar = grad_U(U, t) @ V
        return jnp.concatenate([rhs(U, t), V_bar.flatten()], axis=0)

    t0, tf, N = times[:3]
    hmax = times[3] if len(times)>3 else 1e-2
    N_ = N//nb_splits + 1

    nb_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((nb_devices, 1))
    shard = sharding.PositionalSharding(devices)

    n_s = np.arange(nb_splits)
    t0_s = t0 + n_s*(tf-t0)/nb_splits
    tf_s = t0 + (n_s+1)*(tf-t0)/nb_splits
    t_s = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, 2), in_axes=(0, 0))(t0_s, tf_s)
    t_s = jax.device_put(t_s, shard)

    def cond_fun(carry):
        _, _, errors, k = carry
        return (errors[k] > tol) & (k<max_iter)

    def body_fun(carry):
        _, U, errors, k = carry
  
        V = jnp.broadcast_to(jnp.diag(jnp.ones(nz)).flatten(), (nb_splits, nz*nz))
        UV0_s = jnp.concatenate([U[:-1,:], V], axis=1)
        UV0_s = jax.device_put((UV0_s), shard)

        UVf = jax.vmap(integrator, in_axes=(None, None, 0, 0, None, None, None, None, None, None))(aug_rhs, static, UV0_s, t_s, 1e-1, 1e-1, jnp.inf, 20, 2, "checkpointed")[:, -1, ...]

        def step(U_kp1_n, n):
            V_prev = UVf[n-1, nz:].reshape((nz, nz))
            U_kp1_np1 = UVf[n-1, :nz] + V_prev@(U_kp1_n - U[n-1, :])
            return (U_kp1_np1), U_kp1_np1

        _, U_next = jax.lax.scan(step, (z0), n_s+1)

        U_next = jnp.concatenate([z0[None, :], U_next[:, :]], axis=0)
        errors = errors.at[k+1].set(jnp.linalg.norm(U_next - U))

        return U, U_next, errors, k+1

    errors = jnp.inf * jnp.ones((max_iter+1,))
    errors = errors.at[0].set(2*tol)

    B0 = jnp.reshape(B0, (B0.shape[0], -1))     ## TODO Carefull how it orders the dimensions
    _, U_star, errors, nb_iters = eqx.internal.while_loop(cond_fun, body_fun, (B0, B0, errors, 0), max_steps=2, kind="checkpointed")

    return jnp.reshape(U_star, (U_star.shape[0], *orig_shape)), errors[1:], nb_iters




# def parareal(func, B0, z0, nb_splits, times, rhs_params, static, integrator, learning_rate, tol, max_iter):

#     coarse_integrator = euler_integrator
#     fine_integrator = integrator

#     t0, tf, N = times[:3]
#     hmax = times[3] if len(times)>3 else 1e-2
#     N_ = N//nb_splits + 1

#     nb_devices = jax.local_device_count()
#     devices = mesh_utils.create_device_mesh((nb_devices, 1))
#     shard = sharding.PositionalSharding(devices)

#     n_s = jnp.arange(nb_splits)
#     t0_s = t0 + n_s*(tf-t0)/nb_splits
#     tf_s = t0 + (n_s+1)*(tf-t0)/nb_splits
#     t_s = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, N_), in_axes=(0, 0))(t0_s, tf_s)
#     t_s_pr = jax.device_put((t_s), shard)

#     def cond_fun(carry):
#         _, _, errors, k = carry
#         return (errors[k] > tol) & (k<max_iter)

#     def body_fun(carry):
#         _, U, errors, k = carry

#         U_pr = jax.device_put((U[:-1, :]), shard)
#         Uf = jax.vmap(fine_integrator, in_axes=(None, None, 0, 0, None))(rhs_params, static, U_pr, t_s_pr, hmax)[:, -1, ...]

#         # U_prevprev = jax.vmap(coarse_integrator, in_axes=(None, None, 0, 0, None))(rhs_params, static, U[:-1,:], t_s, np.inf)[:, -1, :]
#         U_prevprev = jax.vmap(coarse_integrator, in_axes=(None, None, 0, 0, None))(rhs_params, static, U_pr, t_s_pr, np.inf)[:, -1, :]

#         def step(U_kp1_n, n):
#             U_prev_n = coarse_integrator(rhs_params, static, U_kp1_n, t_s[n-1], np.inf)[-1, :]
#             U_kp1_np1 = Uf[n-1, :] + U_prev_n - U_prevprev[n-1, :]
#             return (U_kp1_np1), U_kp1_np1

#         _, U_next = jax.lax.scan(step, (z0[:]), n_s+1)

#         U_next = jnp.concatenate([z0[None, :], U_next[:, :]], axis=0)
#         errors = errors.at[k+1].set(jnp.linalg.norm(U_next - U))

#         ### TODO Atempt to avoid update U_km1
#         # _, U_next = jax.lax.scan(step, (Uf[k, :]), n_s[k:]+1)
#         # U_sol = jnp.concatenate([U[:k, :], U_next[:, :]], axis=0)
#         # errors = errors.at[k+1].set(jnp.linalg.norm(U_sol - U))
#         # return U, U_sol, errors, k+1

#         return U, U_next, errors, k+1

#     errors = jnp.inf * jnp.ones((max_iter+1,))
#     errors = errors.at[0].set(2*tol)

#     _, U_star, errors, nb_iters = jax.lax.while_loop(cond_fun, body_fun, (B0, B0, errors, 0))

#     return U_star, errors[1:], nb_iters







def parareal(func, B0, z0, nb_splits, times, rhs_params, static, integrator, learning_rate, tol, max_iter):

    coarse_integrator = euler_integrator
    fine_integrator = integrator

    orgi_shape = z0.shape
    # z0 = z0.flatten()

    t0, tf, N = times[:3]
    hmax = times[3] if len(times)>3 else 1e-2
    N_ = N//nb_splits + 1

    nb_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((nb_devices, 1))
    shard = sharding.PositionalSharding(devices)

    n_s = jnp.arange(nb_splits)
    t0_s = t0 + n_s*(tf-t0)/nb_splits
    tf_s = t0 + (n_s+1)*(tf-t0)/nb_splits

    t_s_cr = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, N_), in_axes=(0, 0))(t0_s, tf_s)

    t_s = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, 2), in_axes=(0, 0))(t0_s, tf_s)
    t_s_pr = jax.device_put((t_s), shard)

    def cond_fun(carry):
        _, _, errors, k = carry
        return (errors[k] > tol) & (k<max_iter)

    def body_fun(carry):
        _, U, errors, k = carry

        U_pr = jax.device_put((U[:-1, :]), shard)
        Uf = jax.vmap(fine_integrator, in_axes=(None, None, 0, 0, None, None, None, None, None, None))(rhs_params, static, U_pr, t_s_pr, 1e-1, 1e-1, jnp.inf, 20, 2, "checkpointed")[:, -1, ...] ### This defeats the purpose of NodePinT doesn't it !

        # U_prevprev = jax.vmap(coarse_integrator, in_axes=(None, None, 0, 0, None))(rhs_params, static, U[:-1,:], t_s, np.inf)[:, -1, :]
        U_prevprev = jax.vmap(coarse_integrator, in_axes=(None, None, 0, 0, None, None, None, None, None, None))(rhs_params, static, U_pr, t_s_cr, 1e-1, 1e-1, jnp.inf, 20, 2, "checkpointed")[:, -1, :]

        def step(U_kp1_n, n):
            U_prev_n = coarse_integrator(rhs_params, static, U_kp1_n, t_s[n-1], 1e-1, 1e-1, jnp.inf, 20, 2, "checkpointed")[-1, :]
            U_kp1_np1 = Uf[n-1, :] + U_prev_n - U_prevprev[n-1, :]
            return (U_kp1_np1), U_kp1_np1

        _, U_next = jax.lax.scan(step, (z0[:]), n_s+1)

        U_next = jnp.concatenate([z0[None, :], U_next[:, :]], axis=0)
        errors = errors.at[k+1].set(jnp.linalg.norm(U_next - U))

        ### TODO Atempt to avoid update U_km1
        # _, U_next = jax.lax.scan(step, (Uf[k, :]), n_s[k:]+1)
        # U_sol = jnp.concatenate([U[:k, :], U_next[:, :]], axis=0)
        # errors = errors.at[k+1].set(jnp.linalg.norm(U_sol - U))
        # return U, U_sol, errors, k+1

        return U, U_next, errors, k+1

    errors = jnp.inf * jnp.ones((max_iter+1,))
    errors = errors.at[0].set(2*tol)

    # B0 = jnp.reshape(B0, (B0.shape[0], -1))
    _, U_star, errors, nb_iters = eqx.internal.while_loop(cond_fun, body_fun, (B0, B0, errors, 0), max_steps=2, kind="checkpointed")

    # U_star = jnp.reshape(U_star, (U_star.shape[0], *orgi_shape))
    return U_star, errors[1:], nb_iters

















## TODO anderson acceleration from http://implicit-layers-tutorial.org/implicit_functions/
# def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
#   x0 = z_init
#   x1 = f(x0)
#   x2 = f(x1)
#   X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
#   F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

#   def step(n, k, X, F):
#     G = F[:n] - X[:n]
#     GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
#     H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
#                    [ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
#     alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]

#     xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n])
#     X = X.at[k % m].set(xk)
#     F = F.at[k % m].set(f(xk))
#     return X, F

#   # unroll the first m steps
#   for k in range(2, m):
#     X, F = step(k, k, X, F)
#     res = jnp.linalg.norm(F[k] - X[k]) / (1e-5 + jnp.linalg.norm(F[k]))
#     if res < tol or k + 1 >= max_iter:
#       return X[k], k

#   # run the remaining steps in a lax.while_loop
#   def body_fun(carry):
#     k, X, F = carry
#     X, F = step(m, k, X, F)
#     return k + 1, X, F

#   def cond_fun(carry):
#     k, X, F = carry
#     kmod = (k - 1) % m
#     res = jnp.linalg.norm(F[kmod] - X[kmod]) / (1e-5 + jnp.linalg.norm(F[kmod]))
#     return (k < max_iter) & (res >= tol)

#   k, X, F = lax.while_loop(cond_fun, body_fun, (k + 1, X, F))
#   return X[(k - 1) % m]











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

    B_star, errors, nb_iters = pint_scheme(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)

    return B_star, errors, nb_iters


def fixed_point_ad_fwd(func, B0, z0, nb_splits, times, rhs, static, integrator, pint_scheme, learning_rate, tol, max_iter):
    ## TODO should I add an argument "pint_scheme" for the fixed point/root function ?

    # B_star = fixed_point_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)
    # B_star = direct_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)
    # B_star = newton_root_finder(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)

    B_star, errors, nb_iters = pint_scheme(func, B0, z0, nb_splits, times, rhs, static, integrator, learning_rate, tol, max_iter)

    return (B_star, errors, nb_iters), (B_star, z0, rhs)


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

    w = inner_fixed_point_ad(fp_func, B_star, z0, nb_splits, times, rhs, static, integrator, v[0], learning_rate, tol, max_iter)

    theta_bar, = vjp_theta(w)

    return jnp.zeros_like(B_star), None, theta_bar


fixed_point_ad.defvjp(fixed_point_ad_fwd, fixed_point_ad_bwd)
