
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from nodepint.utils import get_key
from nodepint.training import train_parallel_neural_ode
from nodepint.data import load_jax_dataset, convert_to_one_hot_encoding
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
        for i in range(4):
            self.layers = self.layers + [jax.nn.relu, eqx.nn.Linear(100, 100, key=key)]

        # self.prediction_layer = eqx.nn.Linear(100, 10, key=key)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#%%

ds = load_jax_dataset(path="mnist", split="train")
ds = convert_to_one_hot_encoding(ds, feature="label")


## Optax crossentropy loss
loss = optax.softmax_cross_entropy
optimscheme = optax.adam

## Define the neural ODE
neuralnet = MLP()

## Train the neural ODE
dynamicnet, loss_hts = train_parallel_neural_ode(neuralnet,
                                    ds,
                                    pint_scheme="newton",
                                    proj_scheme="random",
                                    solver=euler_integrator, 
                                    loss_fn=loss, 
                                    optim_scheme=optimscheme, 
                                    nb_processors=4, 
                                    shooting_learning_rate=1e-3, 
                                    scheduler=optax.constant_schedule(1e-3), 
                                    times=jnp.linspace(0, 1, 1000))
