
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from nodepint.utils import get_key
from nodepint.training import train_parallel_neural_ode
from nodepint.data import load_jax_dataset, convert_to_one_hot_encoding
from nodepint.integrators import dopri_integrator

class MLP(eqx.Module):
    """
    A simple neural net that learn MNIST
    """

    hidden_layers: list

    prediction_layer: eqx.nn.Linear

    def __init__(self, key=None):

        key = get_key(key)

        self.hidden_layers = []
        for i in range(4):
            self.hidden_layers = self.hidden_layers + [eqx.nn.Linear(100, 100, key=key), jax.nn.relu]

        self.prediction_layer = eqx.nn.Linear(100, 10, key=key)

    def __call__(self, x, t):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def predict(self, x):
        return self.prediction_layer(x)

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
                                    pintscheme="newton",
                                    projscheme="random",
                                    solver=dopri_integrator, 
                                    loss=loss, 
                                    optimscheme=optimscheme, 
                                    nb_processors=4, 
                                    deep_learning_rate=1e-3, 
                                    scheduler=optax.constant_schedule(1e-3), times=jnp.linspace(0, 1, 1000))
