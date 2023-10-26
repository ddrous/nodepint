
#%%
import time
import jax

## Use jax cpu
# jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
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
from nodepint.data import load_mnist_dataset_torch
from nodepint.integrators import dopri_integrator, euler_integrator, rk4_integrator, dopri_integrator_diff
from nodepint.pint import newton_root_finder, direct_root_finder, fixed_point_finder, direct_root_finder_aug, parareal
from nodepint.sampling import random_sampling, identity_sampling, neural_sampling

import cProfile

import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'    ## Trick to virtualise CPU for pmap
print("Available devices:", jax.devices())

# os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'    ## For things to work on JADE (single-threaded compilation)

# nb_devices = jax.local_device_count()

SEED = 27


#%% [markdown]
# ## Define neural net

#%%


# class MLP(eqx.Module):
#     """
#     A simple neural net that learn MNIST
#     """

#     layers: list
#     # prediction_layer: eqx.nn.Linear

#     def __init__(self, key=None):

#         key = get_new_keys(key)

#         self.layers = [eqx.nn.Linear(100, 100, key=key)]
#         for i in range(1):
#             self.layers = self.layers + [jax.nn.relu, eqx.nn.Linear(100, 100, key=key)]
#         # self.prediction_layer = eqx.nn.Linear(100, 10, key=key)

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

class Encoder(eqx.Module):
    """
    A convolutional encoder for MNIST
    """
    layers: list

    def __init__(self, key=None):
        keys = get_new_keys(key, num=3)
        self.layers = [eqx.nn.Conv2d(1, 64, (3, 3), stride=1, key=keys[0]), jax.nn.relu, eqx.nn.GroupNorm(64, 64),
                        eqx.nn.Conv2d(64, 64, (4, 4), stride=2, padding=1, key=keys[1]), jax.nn.relu, eqx.nn.GroupNorm(64, 64),
                        eqx.nn.Conv2d(64, 4, (4, 4), stride=2, padding=1, key=keys[2]) ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




class Processor(eqx.Module):
    """
    A convlutional processor to be passed to the neural ODE
    """

    layers: list

    def __init__(self, key=None):
        keys = get_new_keys(key, num=2)
        self.layers = [eqx.nn.Conv2d(4+1, 64, (3, 3), stride=1, padding=1, key=keys[0]), jax.nn.tanh,
                        eqx.nn.Conv2d(64, 4, (3, 3), stride=1, padding=1, key=keys[1]), jax.nn.tanh]

    def __call__(self, x, t):
        y = jnp.concatenate([jnp.broadcast_to(t, (1,)+x.shape[1:]), x], axis=0)
        for layer in self.layers:
            y = layer(y)
        return y




class Decoder(eqx.Module):
    """
    A decoder to classify MNIST
    """

    layers: list

    def __init__(self, key=None):
        key = get_new_keys(key, 1)
        # self.layers = [eqx.nn.GroupNorm(64, 64), jax.nn.relu, 
        #                 eqx.nn.AvgPool2d((6, 6)), lambda x:jnp.reshape(x, (64,)),
        #                 eqx.nn.Linear(64, 10, key=key)]
        self.layers = [eqx.nn.GroupNorm(4, 4), jax.nn.relu, 
                        eqx.nn.AvgPool2d((6, 6)), lambda x:jnp.reshape(x, (4,)),
                        eqx.nn.Linear(4, 10, key=key)]


    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x





#%% [markdown]
# ## Load the dataset

#%%

ds = load_mnist_dataset_torch(root="./data/mnist", train=True)
# ds = make_dataloader_torch(ds, subset_size="all", seed=SEED, norm_factor=255.)

# print("Feature names:", get_dataset_features(ds))
print("Number of training examples:", len(ds))

## Visualise a datapoint
np.random.seed(time.time_ns()%(2**32))
point_id = np.random.randint(0, len(ds))
pixels, label = ds[point_id]

plt.title(f"Label is {label:1d}")
plt.imshow(pixels.squeeze(), cmap='gray')
plt.show()

# import torch
# print("Torch devices out there", torch.cuda.device_count())


#%% [markdown]
# ## Define training parameters

#%%

## Optax crossentropy loss
optim_scheme = optax.adam
# times = tuple(np.linspace(0, 1, 101).flatten())
times = (0., 1., 11)       ## t0, tf, nb_times (this is for solving the ODE if an adaptative time stepper is not used. Not for eval)

integrator_args = (1e-1, 1e-1, jnp.inf, 20, 2, "checkpointed")     ## rtol, atol, max_dt, max_steps, kind, max_steps_rev (these are typically by adatative time steppers)
fixed_point_args = (1., 1e-6, 5)               ## learning_rate, tol, max_iter

# loss = optax.softmax_cross_entropy
loss = optax.softmax_cross_entropy_with_integer_labels

# def cross_entropy_fn(y_pred, y):      ## TODO: should be vmapped by design
#     y_pred = jnp.argmax(jax.nn.softmax(y_pred, axis=-1), axis=-1)
#     y = jnp.argmax(y, axis=-1)

#     return jnp.mean(y_pred == y, axis=-1)


## Base neural ODE model
# neuralnet = MLP(key=SEED)
# neuralnet = eqx.nn.MLP(in_size=100, out_size=100, width_size=250, depth=3, activation=jax.nn.relu, key=get_key(None))

keys = get_new_keys(SEED, num=3)
neural_nets = (Encoder(key=keys[0]), Processor(key=keys[1]), Decoder(key=keys[2]))

## PinT scheme with only mandatory arguments

nb_epochs = 5
batch_size = 120*1       ## Divisible by the dataset size to avoid recompilation !
total_steps = nb_epochs*(len(ds)//batch_size)

scheduler = optax.piecewise_constant_schedule(init_value=1e-4, boundaries_and_scales={int(total_steps*0.5):0.25, int(total_steps*0.75):0.25})


key = get_new_keys(SEED)

train_params = {"neural_nets":neural_nets,
                "data":ds,
                # "pint_scheme":fixed_point_finder,
                # "pint_scheme":direct_root_finder_aug,
                "pint_scheme":parareal,
                "samp_scheme":neural_sampling,
                # "samp_scheme":identity_sampling,
                # "integrator":rk4_integrator, 
                # "integrator":euler_integrator, 
                "integrator":dopri_integrator,
                # "integrator":dopri_integrator_diff,
                "integrator_args":integrator_args,
                "loss_fn":loss,
                "optim_scheme":optim_scheme, 
                "nb_processors":4,
                "scheduler":scheduler,
                "times":times,
                "fixed_point_args":fixed_point_args,
                "nb_epochs":nb_epochs,
                "batch_size":batch_size,
                "repeat_projection":1,
                "nb_vectors":5,
                "force_serial":True,
                "key":key}


#%% [markdown]
# ## Train the model

#%% 

# with jax.profiler.trace("./runs", create_perfetto_link=False):


# profiler = cProfile.Profile()
# profiler.enable()

start_time = time.time()
cpu_start_time = time.process_time()

trained_networks, shooting_fn, loss_hts, errors_hts, nb_iters_hts = train_project_neural_ode(**train_params)

clock_time = time.process_time() - cpu_start_time
wall_time = time.time() - start_time

# print("\nNumber of iterations till PinT eventual convergence:\n", np.asarray(nb_iters_hts))
# print("Errors during PinT iterations:\n", np.asarray(errors_hts))

time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal training time: %d hours %d mins %d secs" %time_in_hmsecs)

# profiler.disable()
# # profiler.print_stats(sort='cumulative')
# profile_output_filename = "runs/cprofile/profile_report.txt"
# with open(profile_output_filename, "w") as f:
#     profiler.dump_stats(profile_output_filename)

#%% 
# eqx.tree_serialise_leaves("data/encode_process_decode.eqx", neural_nets)

#%% 
# trained_networks = eqx.tree_deserialise_leaves("data/encode_process_decode.eqx", neural_nets)
# shooting_fn, loss_hts, errors_hts, nb_iters_hts = [None]*4

#%% [markdown]
# ## Analyse loss history

#%% 

## Plot the loss histories per iterations
labels = [str(i) for i in range(len(loss_hts))]
epochs = range(len(loss_hts[0]))

sbplot(epochs, jnp.stack(loss_hts, axis=-1), label=labels, x_label="epochs", y_scale="log", title="Loss histories");

## Loss histories acros all iterations
total_loss = np.concatenate(loss_hts, axis=0)
total_epochs = 1 + np.arange(len(total_loss))

ax = sbplot(total_epochs, total_loss, x_label="epochs", y_scale="log", title="Total loss history");






# #%% [markdown]
# # ## Quick profiling

# #%% 
# import cProfile

# def main():
#     train_parallel_neural_ode(**train_params)

# cProfile.run('main()', sort='cumtime')


#%% [markdown]
# ## Compute metrics on a test dataset

#%% 

## Load the test dataset
test_ds = load_mnist_dataset_torch(root="./data/mnist", train=False)

print("\nNumber of testing examples", len(test_ds))


def accuracy_fn(y_pred, y):
    y_pred = jnp.argmax(jax.nn.softmax(y_pred, axis=-1), axis=-1)
    # y = jnp.argmax(y, axis=-1)

    return jnp.sum(y_pred == y, axis=-1)*100


test_params = {"neural_nets": trained_networks,
                "data":test_ds,
                "pint_scheme":fixed_point_finder,       ## If None then the fixed_point_ad_rule is used
                # "pint_scheme":direct_scheme,
                # "integrator":rk4_integrator,
                "integrator":dopri_integrator_diff,
                "integrator_args":integrator_args,
                "fixed_point_args":fixed_point_args,
                "acc_fn":accuracy_fn,
                "shooting_fn":shooting_fn,
                "nb_processors":16,
                "times":times,
                "batch_size":120}


start_time = time.time()

avg_acc = test_neural_ode(**test_params)

test_wall_time = time.time() - start_time
time_in_hms= seconds_to_hours(test_wall_time)

print(f"\nAverage accuracy: {avg_acc:.2f} %")
print("Test time: %d hours %d mins %d secs" %time_in_hms)

# #%% [markdown]
# # ## Write stuff to tensorboard

# #%% 

# run_name = str(datetime.datetime.now().strftime("%H:%M %d-%m-%Y"))[:19]
# writer = tensorboard.SummaryWriter("runs/"+run_name)

# hps = {}

# hps["bach_size"] = train_params["batch_size"]
# hps["scheduler"] = train_params["scheduler"]
# hps["nb_epochs"] = train_params["nb_epochs"]
# hps["nb_processors"] = train_params["nb_processors"]
# hps["repeat_projection"] = train_params["repeat_projection"]
# hps["nb_vectors"] = train_params["nb_vectors"]

# hps["times"] = (train_params["times"][0], train_params["times"][-1], len(train_params["times"]))
# hps["optim_scheme"] = train_params["optim_scheme"].__name__
# hps["pint_scheme"] = str(train_params["pint_scheme"])[45:-63]
# hps["key"] = SEED
# hps["integrator"] = train_params["integrator"].__name__
# hps["loss_fn"] = train_params["loss_fn"].__name__
# hps["data"] = str(get_dataset_features(train_params["data"]))
# hps["dynamicnet_size"] = sum(x.size for x in jax.tree_util.tree_leaves(eqx.partition(dynamicnet, eqx.is_array)[0]))

# hps["wall_time"] = wall_time
# hps["clock_time"] = clock_time

# hps["test_acc"] = avg_acc
# hps["test_wall_time"] = test_wall_time

# writer.hparams(hps)


# for ep in range(len(total_epochs)):
#     writer.scalar('train_loss', total_loss[ep], ep+1)

# writer.flush()
# writer.close()

# %%
