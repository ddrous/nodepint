
#%%
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from nodepint.utils import get_key, sbplot
from nodepint.training import train_parallel_neural_ode
from nodepint.data import load_jax_dataset, convert_to_one_hot_encoding, normalise_feature
from nodepint.integrators import dopri_integrator, euler_integrator

class MLP(eqx.Module):
    """
    A simple neural net that learn MNIST
    """

    layers: list

    # prediction_layer: eqx.nn.Linear

    def __init__(self, key=None):

        key = get_key(key)

        self.layers = [eqx.nn.Linear(100, 100, key=key)]
        for i in range(2):
            self.layers = self.layers + [jax.nn.relu, eqx.nn.Linear(100, 100, key=key)]

        # self.prediction_layer = eqx.nn.Linear(100, 10, key=key)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#%%

ds = load_jax_dataset(path="mnist", split="train")
ds = convert_to_one_hot_encoding(ds, feature="label")
ds = normalise_feature(ds, feature="image", factor=255.)
print("ds", ds, "features", ds.features)

## Visualise the dataset

pixels = ds[0]["image"]
label = ds[0]["label"]

# Plot
plt.title('Label is {label}'.format(label=label.argmax()))
plt.imshow(pixels, cmap='gray')
plt.show()

# print(pixels)

#%%
## Optax crossentropy loss
loss = optax.softmax_cross_entropy
optimscheme = optax.adam
times = tuple(np.linspace(0, 1, 101).flatten())
# times = 1

## Define the neural ODE
neuralnet = MLP()

## Train the neural ODE
dynamicnet, basis, loss_hts = train_parallel_neural_ode(neuralnet,
                                    ds,
                                    pint_scheme="newton",
                                    # pint_scheme="direct",
                                    proj_scheme="random",
                                    integrator=euler_integrator, 
                                    # solver=dopri_integrator, 
                                    loss_fn=loss, 
                                    optim_scheme=optimscheme, 
                                    nb_processors=4, 
                                    shooting_learning_rate=1e-3, 
                                    scheduler=optax.constant_schedule(1e-3),
                                    times=times,
                                    nb_epochs=100)


#%%

# dynamicnet
loss_hts

## Plot the loss histories accross iterations
# sbplot(jnp.concatenate(loss_hts), title="Loss history")

labels = [str(i) for i in range(len(loss_hts))]
epochs = range(len(loss_hts[0]))

sbplot(epochs, jnp.stack(loss_hts, axis=-1), label=labels, x_label="epochs", title="Loss history");
