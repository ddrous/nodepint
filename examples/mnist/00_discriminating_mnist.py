
#%%
import time
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from functools import partial
import datetime
from flax.metrics import tensorboard

from nodepint.utils import get_new_keys, sbplot, seconds_to_hours
from nodepint.training import train_parallel_neural_ode, test_dynamic_net
from nodepint.data import load_jax_dataset, get_dataset_features, preprocess_mnist
from nodepint.integrators import dopri_integrator, euler_integrator
from nodepint.pint import newton_root_finder, direct_root_finder
from nodepint.projection import random_sampling, identity_sampling

## Use jax cpu
# jax.config.update("jax_platform_name", "cpu")

SEED = 27


#%% [markdown]
# ## Define neural net

#%%

class MLP(eqx.Module):
    """
    A simple neural net that learn MNIST
    """

    layers: list
    # prediction_layer: eqx.nn.Linear

    def __init__(self, key=None):

        key = get_new_keys(key)

        self.layers = [eqx.nn.Linear(100, 100, key=key)]
        for i in range(1):
            self.layers = self.layers + [jax.nn.relu, eqx.nn.Linear(100, 100, key=key)]
        # self.prediction_layer = eqx.nn.Linear(100, 10, key=key)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



#%% [markdown]
# ## Load the dataset

#%%

ds = load_jax_dataset(path="mnist", split="train")
ds = preprocess_mnist(ds, subset_size=32, seed=SEED, norm_factor=255.)

print("features", get_dataset_features(ds))

## Visualise a datapoint
pixels = ds[0]["image"]
label = ds[0]["label"]

plt.title('Label is {label}'.format(label=label.argmax()))
plt.imshow(pixels, cmap='gray')
plt.show()



#%% [markdown]
# ## Define training parameters

#%%

## Optax crossentropy loss
optim_scheme = optax.adam
times = tuple(np.linspace(0, 1, 101).flatten())

loss = optax.softmax_cross_entropy

# def cross_entropy_fn(y_pred, y):      ## TODO: should be vmapped by design
#     y_pred = jnp.argmax(jax.nn.softmax(y_pred, axis=-1), axis=-1)
#     y = jnp.argmax(y, axis=-1)

#     return jnp.mean(y_pred == y, axis=-1)


## Base neural ODE model
neuralnet = MLP(key=SEED)
# neuralnet = eqx.nn.MLP(in_size=100, out_size=100, width_size=250, depth=3, activation=jax.nn.relu, key=get_key(None))

## PinT scheme with only mandatory arguments
# newton_scheme = partial(newton_root_finder, learning_rate=1., tol=1e-6, max_iter=3)

# direct_scheme = partial(direct_root_finder, learning_rate=1., tol=1e-6, max_iter=3)
direct_scheme = direct_root_finder

key = get_new_keys(SEED)

train_params = {"neural_net":neuralnet,
                "data":ds,
                "pint_scheme":direct_scheme,
                "proj_scheme":random_sampling,
                # "proj_scheme":identity_sampling,
                "integrator":euler_integrator, 
                # "integrator":dopri_integrator, 
                "loss_fn":loss, 
                "optim_scheme":optim_scheme, 
                "nb_processors":4, 
                "scheduler":1e-3,
                "times":times,
                "nb_epochs":20,
                "batch_size":16,
                "repeat_projection":2,
                "nb_vectors":4,
                "key":key}


#%% [markdown]
# ## Train the model

#%% 


start_time = time.time()
cpu_start_time = time.process_time()

dynamicnet, basis, shooting_fn, loss_hts = train_parallel_neural_ode(**train_params)

clock_time = time.process_time() - cpu_start_time
wall_time = time.time() - start_time

time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal training time: %d hours %d mins %d secs" %time_in_hmsecs)



#%% [markdown]
# ## Analyse loss history

#%% 

## Plot the loss histories per iterations
labels = [str(i) for i in range(len(loss_hts))]
epochs = range(len(loss_hts[0]))

sbplot(epochs, jnp.stack(loss_hts, axis=-1), label=labels, x_label="epochs", title="Loss history");

## Loss histories acros all iterations
total_loss = np.concatenate(loss_hts, axis=0)
total_epochs = 1 + np.arange(len(total_loss))

ax = sbplot(total_epochs, total_loss, x_label="epochs", title="Total loss history");




# #%% [markdown]
# # ## Compute metrics on a test dataset

# #%% 

# ## Load the test dataset
# test_ds = load_jax_dataset(path="mnist", split="test")
# test_ds = preprocess_mnist(test_ds, subset_size=1280, seed=SEED, norm_factor=255.)


# def accuracy_fn(y_pred, y):
#     y_pred = jnp.argmax(jax.nn.softmax(y_pred, axis=-1), axis=-1)
#     y = jnp.argmax(y, axis=-1)

#     return jnp.mean(y_pred == y, axis=-1)*100

# test_params = {"neural_net":dynamicnet,
#                 "data":test_ds,
#                 "basis":basis,
#                 "pint_scheme":newton_scheme,
#                 "integrator":euler_integrator, 
#                 "acc_fn":accuracy_fn, 
#                 "shooting_fn":shooting_fn,
#                 "nb_processors":4, 
#                 "times":times,
#                 "batch_size":8}


# start_time = time.time()

# avg_acc = test_dynamic_net(**test_params)

# test_wall_time = time.time() - start_time
# time_in_hms= seconds_to_hours(test_wall_time)

# print(f"\nAverage accuracy: {avg_acc:.2f} %")
# print("Test time: %d hours %d mins %d secs" %time_in_hms)

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
