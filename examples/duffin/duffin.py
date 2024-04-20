
#%%
import time

## Use jax cpu
# jax.config.update("jax_platform_name", "cpu")

## Limit JAX memory usage
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

## Jax debug nans
jax.config.update("jax_debug_nans", True)

import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from functools import partial
import datetime
# from flax.metrics import tensorboard

from nodepint.utils import get_new_keys, sbplot, seconds_to_hours
from nodepint.training import train_project_neural_ode, test_neural_ode
# from nodepint.data import load_jax_dataset, get_dataset_features, preprocess_mnist
from nodepint.data import load_mnist_dataset_torch, load_lotka_volterra_dataset, ODEDataset
from nodepint.integrators import dopri_integrator, euler_integrator, rk4_integrator, dopri_integrator_diff, dopri_integrator_diffrax, euler_integrator_duffin
from nodepint.pint import newton_root_finder, direct_root_finder, fixed_point_finder, direct_root_finder_aug, parareal
from nodepint.sampling import random_sampling, identity_sampling, neural_sampling

import cProfile

import os
print("Available devices:", jax.devices())
import warnings
warnings.filterwarnings("ignore")

SEED = 2026

## Reload the nodepint package before each cell run
%load_ext autoreload
%autoreload 2

#%% [markdown]
# ## Define neural net

#%%



class Encoder(eqx.Module):
    # layers: list

    #### Use a tensordot, and sum over all the three/two later dimensions of the model. 
    ## If basis = (2,4,4, 1,28,28) and x of shape (1,28,28) then the tensordot will
    ## will return something of shape (2,4,4)
    ## Finally I can multiply this by a learned weight of shape (2,4,4) as well

    def __init__(self, key=None):
        # keys = get_new_keys(key, num=3)
        # self.layers = [eqx.nn.Conv2d(1, 64, (3, 3), stride=1, key=keys[0]), jax.nn.relu, eqx.nn.GroupNorm(64, 64),
        #                 eqx.nn.Conv2d(64, 64, (4, 4), stride=2, padding=1, key=keys[1]), jax.nn.relu, eqx.nn.GroupNorm(64, 64),
        #                 eqx.nn.Conv2d(64, 64, (4, 4), stride=2, padding=1, key=keys[2]) ]
        pass

    def __call__(self, x):
        # for layer in self.layers:
        #     x = layer(x)
        return x


class Processor(eqx.Module):
    layers: list

    def __init__(self, key=None):
        keys = get_new_keys(key, num=4)
        self.layers = [eqx.nn.Linear(2, 64, key=keys[0]), 
                       jax.nn.softplus, 
                       eqx.nn.Linear(64, 64, key=keys[1]),
                       jax.nn.softplus,
                       eqx.nn.Linear(64, 64, key=keys[3]),
                       jax.nn.softplus,
                       eqx.nn.Linear(64, 2, key=keys[2])]

    def __call__(self, x, t):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)+x.shape[1:]), x], axis=0)
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class Decoder(eqx.Module):

    # layers: list

    def __init__(self, key=None):
        # key = get_new_keys(key, 1)
        # self.layers = [eqx.nn.GroupNorm(64, 64), jax.nn.relu, 
        #                 eqx.nn.AvgPool2d((6, 6)), lambda x:jnp.reshape(x, (64,)),
        #                 eqx.nn.Linear(64, 10, key=key)]
        # # self.layers = [eqx.nn.GroupNorm(4, 4), jax.nn.relu, 
        # #                 eqx.nn.AvgPool2d((6, 6)), lambda x:jnp.reshape(x, (4,)),
        # #                 eqx.nn.Linear(4, 10, key=key)]
        pass

    def __call__(self, x):
        # for layer in self.layers:
        #     x = layer(x)
        return x


class DuffinPhysics(eqx.Module):
    params: jnp.ndarray
    def __init__(self, key=None):
        keys = get_new_keys(key, 3)
        ## a=-1/2., b=-1., c=1/10.
        a = jax.random.uniform(keys[0], (1,), minval=-0.75, maxval=-0.25)[0]
        b = jax.random.uniform(keys[1], (1,), minval=-1.5, maxval=-0.5)[0]
        c = jax.random.uniform(keys[2], (1,), minval=0.05, maxval=0.15)[0]
        self.params = jnp.array([a, b, c])

    def __call__(self, y, t):
        a, b, c = self.params
        x, y = y

        dxdt = y
        dydt = a*y - x*(b + c*x**2)
        return jnp.array([dxdt, dydt])




#%% [markdown]
# ## Load the dataset

#%%

ds = ODEDataset(root_dir="./data", split="train", skip=10)
# ds = make_dataloader_torch(ds, subset_size="all", seed=SEED, norm_factor=255.)

print("Number of training examples:", len(ds))

## Visualise a datapoint
np.random.seed(time.time_ns()%(2**32))
point_id = np.random.randint(0, len(ds))
init_cond, trajectory = ds[point_id]
t_eval = ds.t

# ## Set figure size
# plt.figure(figsize=(8, 4))
# plt.title(f"Sample trajectory id: {point_id}")
# plt.plot(t_eval, trajectory[:, 0], label="Displacement")
# plt.plot(t_eval, trajectory[:, 1], label="Velocity")
# plt.show();


## Plot the phase space for all the datapoints
plt.figure(figsize=(8, 4))
plt.title("Phase space")
for i in range(len(ds)):
    init_cond, trajectory = ds[i]
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Trajectory {i}")
plt.show();



#%% [markdown]
# ## Define training parameters

#%%

## Optax crossentropy loss
optim_scheme = optax.adam
# times = tuple(np.linspace(0, 1, 101).flatten())
times = (t_eval[0], t_eval[-1], t_eval.shape[0])       ## t0, tf, nb_times (this is for solving the ODE if an adaptative time stepper is not used. Not for eval)

fine_integrator_args = (1e-8, 1e-8, jnp.inf, 200, 10, "checkpointed")     ## rtol, atol, max_dt, max_steps, kind, max_steps_rev (these are typically by adatative time steppers)
fixed_point_args = (1., 1e-6, 2000)               ## learning_rate, tol, max_iter

coarse_integrator_args = (1e-1, 1e-1, jnp.inf, 20, 2, "checkpointed")

# loss = optax.softmax_cross_entropy
# loss = optax.softmax_cross_entropy_with_integer_labels
loss = optax.l2_loss

keys = get_new_keys(SEED, num=4)
neural_nets = (Encoder(key=keys[0]), Processor(key=keys[1]), Decoder(key=keys[2]), DuffinPhysics(key=keys[3]))

## PinT scheme with only mandatory arguments

nb_epochs = 1500
batch_size = ds.X.shape[0]
total_steps = nb_epochs*(len(ds)//batch_size)

scheduler = optax.piecewise_constant_schedule(init_value=1e-3, boundaries_and_scales={int(total_steps*0.5):0.75, int(total_steps*0.75):0.75})


key = get_new_keys(SEED)





#%% [markdown]
# ## Train the model

train_params = {"neural_nets":neural_nets,
                "data":ds,
                # "pint_scheme":fixed_point_finder,
                # "pint_scheme":direct_root_finder_aug,
                "pint_scheme":parareal,
                "samp_scheme":identity_sampling,
                "coarse_integrator":euler_integrator,
                "coarse_integrator_args":coarse_integrator_args,
                "fine_integrator":rk4_integrator,
                # "fine_integrator":dopri_integrator_diff,
                # "fine_integrator":dopri_integrator_diffrax,
                "fine_integrator_args":fine_integrator_args,
                "loss_fn":loss,
                "optim_scheme":optim_scheme, 
                "nb_processors":len(t_eval)-1,
                "scheduler":scheduler,
                "times":times,
                "fixed_point_args":fixed_point_args,
                "nb_epochs":nb_epochs,
                "batch_size":batch_size,
                "repeat_projection":1,
                "nb_vectors":5,
                "force_serial":False,
                "key":key}

start_time = time.time()
cpu_start_time = time.process_time()

trained_networks, shooting_fn, loss_hts, errors_hts, nb_iters_hts = train_project_neural_ode(**train_params)

clock_time = time.process_time() - cpu_start_time
wall_time = time.time() - start_time

# print("\nNumber of iterations till PinT eventual convergence:\n", np.asarray(nb_iters_hts))
# print("Errors during PinT iterations:\n", np.asarray(errors_hts))

time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal training time: %d hours %d mins %d secs" %time_in_hmsecs)


#%% [markdown]
# ## Analyse loss history

#%% 

# ## Plot the loss histories per iterations
# labels = [str(i) for i in range(len(loss_hts))]
# epochs = range(len(loss_hts[0]))

# sbplot(epochs, jnp.stack(loss_hts, axis=-1), label=labels, x_label="epochs", y_scale="log", title="Loss histories");

## Loss histories acros all iterations
total_loss = np.concatenate(loss_hts, axis=0)
total_epochs = 1 + np.arange(len(total_loss))

ax = sbplot(total_epochs, total_loss, x_label="epochs", y_scale="log", title="Total loss history");


## Print physics paramaters
## a=-1/2., b=-1., c=1/10.
print(f"True physics parameters: a={-0.5}, b={-1}, c={0.1}")
print(f"Learned physics parameters: a={neural_nets[-1].params[0]}, b={neural_nets[-1].params[1]}, c={neural_nets[-1].params[2]}")


## Save the plot
# plt.savefig("loss_history_coarse_euler.png")




#%% [markdown]
# ## Compute metrics on a test dataset

#%% 

## Load the test dataset
test_ds = ODEDataset(root_dir="./data", split="test", skip=10)

print("\nNumber of testing examples", len(test_ds))

def acc_fn(x, y):
    return jnp.mean((x-y)**2)

test_params = {"neural_nets": trained_networks,
                "data":test_ds,
                "pint_scheme":parareal,       ## If None then the fixed_point_ad_rule is used
                # "pint_scheme":direct_scheme,
                "integrator":rk4_integrator,
                # "integrator":dopri_integrator_diffrax,
                "integrator_args":fine_integrator_args,
                "fixed_point_args":fixed_point_args,
                "acc_fn":acc_fn,
                "shooting_fn":shooting_fn,
                "nb_processors":20-1,
                "times":times,
                "batch_size":8}


start_time = time.time()

avg_acc, plot_data = test_neural_ode(**test_params)

print(avg_acc)

test_wall_time = time.time() - start_time
time_in_hms= seconds_to_hours(test_wall_time)

print(f"\nAverage test loss: {avg_acc:.8f}")
print("Test time: %d hours %d mins %d secs" %time_in_hms)



# %%

y0s, ys, y_preds = plot_data
## Plot the phase space for all the datapoints
plt.figure(figsize=(8, 4))
plt.title("Phase space")
colors = ["r", "b", "g", "m", "c", "y", "k", "orange", "purple", "brown", "pink", "gray"]
for i in range(len(ys)):
    if i==0:
        plt.plot(ys[i][:, 0], ys[i][:, 1], label=f"GT Trajectory", color=colors[i], alpha=0.5)
        plt.plot(y_preds[i][:, 0], y_preds[i][:, 1], label=f"Predicted", color=colors[i], linestyle="--", linewidth=3)
    else: ## no label
        plt.plot(ys[i][:, 0], ys[i][:, 1], color=colors[i], alpha=0.5)
        plt.plot(y_preds[i][:, 0], y_preds[i][:, 1], color=colors[i], linestyle="--", linewidth=3)

    ## Massive dot for initial conditions
    plt.scatter(y0s[i][0], y0s[i][1], color=colors[i], s=100)

plt.legend()
plt.xlabel("Displacement")
plt.ylabel("Velocity")
plt.show();
