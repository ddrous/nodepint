#%%

from nodepint import *


class Lorentz(eqx.Module):
    sigma: float
    rho: float
    beta: float

    def __init__(self, sigma=10, rho=28, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def __call__(self, x, t):
        dx = self.sigma * (x[1] - x[0])
        dy = x[0] * (self.rho - x[2]) - x[1]
        dz = x[0] * x[1] - self.beta * x[2]
        return jnp.stack([dx, dy, dz])

rhs = Lorentz()
rhs_params, static = eqx.partition(rhs, eqx.is_array)

fine_integrator = dopri_integrator
# fine_integrator = euler_integrator
func = shooting_function_serial

N = 800 ## nb_processors
lr, tol, max_iters = 1., 1e-6, 20
tf = 5.5
# tf = 5.5
times = (0.0, tf, 10001, 1e-4)

z0 = jnp.array([20., 5., -5.])

B0 = jnp.zeros((N+1, 3)).flatten()
t_init = jnp.linspace(times[0], times[1], N+1)

factor = 25
t_init = increase_timespan(t_init, factor)
B0 = euler_integrator(rhs_params, static, z0, t_init, np.inf)[::factor]

sol, errors, iters = direct_root_finder_aug(func, B0, z0, N, times, rhs_params, static, fine_integrator, lr, tol, max_iters)
# %time sol, errors, iters = parareal(func, B0, z0, N, times, rhs_params, static, fine_integrator, lr, tol, max_iters)

print("Number of iterations:", iters)
print("Errors:", errors)

## Plot the errors
# sbplot(jnp.arange(max_iters), errors, "o-", y_scale="log", x_label="Iteration", y_label="Error", title="Parallel-in-Time Errors");

# jax.debug.visualize_array_sharding(sol)
sol
# print(jax.device_get(sol))      ## Transfer the solution to the CPU



# %%

times_ = jnp.linspace(*times[:3])

# sol_ = fine_integrator(rhs, static, z0, times_, np.inf)[::times[2]//N]
sol_ = dopri_integrator(rhs, static, z0, times_, times[-1])

sol_

# %%


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

ax = sbplot(B0[:, 0], B0[:, 1], B0[:, 2], "r-", lw=1, ax=ax, label="Parallel (initial)")

ax = sbplot(sol_[:, 0], sol_[:, 1], sol_[:, 2], "g-", lw=2, ax=ax, label="Serial")
ax = sbplot(sol[:, 0], sol[:, 1], sol[:, 2], "k--", lw=3, ax=ax, label="Parallel")

ax.view_init(elev=10., azim=255)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor");

## Save this figure to data/lorentz.png
# plt.savefig("data/lorentz.png", dpi=100)

# %%

## Lessons learned:
# - Use parareal for long time horizons, but be ready for slow convergence
# - Use direct root finder for short time horizons, and make sure the initial guess is good enough

# %%

def test():
    ## PinT vs Serial
    assert jnp.allclose(sol, sol_, atol=1e-2) == True

    ## Check that no nans in B0
    assert jnp.all(jnp.isfinite(B0)) == True
