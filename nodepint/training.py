#%%
## Attributes of an ODE and its trajectory for nodepint


import jax
from typing import NamedTuple
from neuralnets import DynamicNeuralNet


# class ODEparams(NamedTuple):
#     times: jax.Array
#     init_state: jax.Array
#     state: jax.Array
#     trajectory: jax.Array

#     # neuralnet: eqx.Module

#     def __init__(self, ...):
#         ## checks that the NN weights match it state shape. If not, do some model surgery




## Dimension-agnostic loss functions for nodepint


## Shape-agnostic optimisers for nodepint




from equinox import Module
from datasets import Dataset

from neuralnets import DynamicNet
from optax import GradientTransformation
from pint import TimeParallelScheme
from projection import ReducedBasisScheme


def train_parallel_neural_ode(neuralnet:Module, data:Dataset, pintmethod:str, projscheme:str, solver:callable, loss:callable, optimiser:GradientTransformation, *args):
    ## Steps in the for loop
    # - Sample a vector
    # - Convert the neural net (of class eqx.Module) into a dynamic one (of class DynamicNet)
    # - Project the data on the current basis
    # - Add neurons to the Dynamic NeuralNet's input layer
    # - Find the solution to that ODE using PinT (after setting the `times` attribute) (the `args` indicate extra parameters for the ODE like the start and end times or times vector)
    # - Upscale the feature vector by multiplying by the transposed basis
    # - Evaluate the (augmented) loss function, and backpropagate


    ## Retruns
    # - The final state (features), trajectory, and readout
    # - The optimised neural network with the spae the user inputted in the begining
    # - The reduced basis
    pass

