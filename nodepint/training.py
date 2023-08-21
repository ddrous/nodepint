#%%
## Attributes of an ODE and its trajectory for nodepint

import jax
import jax.numpy as jnp
from equinox import Module
from datasets import Dataset
import optax
from optax import GradientTransformation

from .neuralnets import (DynamicNet, 
                         add_neurons_to_input_layer, add_neurons_to_output_layer, add_neurons_to_prediction_layer)
from .pint import define_shooting_function, select_root_finding_function
from .projection import select_projection_scheme
from .data import project_dataset_onto_basis, get_dataset_features


def train_parallel_neural_ode(neuralnet:Module, data:Dataset, pintscheme:str, projscheme:str, solver:callable, loss:callable, optimscheme:GradientTransformation, nb_processors=4, *args, **kwargs):
    ## Steps in the for loop
    # - Sample a vector
    # - Convert the neural net (of class eqx.Module) into a dynamic one (of class DynamicNet)
    # - Project the data on the current basis
    # - Add neurons to the Dynamic NeuralNet's input layer
    # - Find the solution to that ODE using PinT (after setting the `times` attribute) (the `args` indicate extra parameters for the ODE like the start and end times or times vector)
    # - Upscale the feature vector by multiplying by the transposed basis
    # - Evaluate the (augmented) loss function, and backpropagate

    ## Get remaiining arguments
    deep_learning_rate = kwargs.get("deep_learning_rate", 1e-3)
    scheduler = kwargs.get("scheduler", optax.constant_schedule(1e-3))
    times = kwargs.get("times", jnp.linspace(0, 1, 1000))

    projscheme = select_projection_scheme(projscheme)
    pintscheme = select_root_finding_function(pintscheme)
    dynamicnet = DynamicNet(neuralnet)

    max_projection_size = 100 ## TODO make this related to the dataset dimension
    basis = None    ## Initialise the basis
    loss_hts = []    ## Initialise the loss history

    for b in range(max_projection_size):

        ## Sample a vector
        data_feature = list(get_dataset_features(data))[-1]
        basis, nb_neurons = projscheme(basis, data[data_feature].shape[1:], key=None)

        ## Add neurons to the Dynamic NeuralNet's input, output, and prediction layers
        dynamicnet = add_neurons_to_input_layer(dynamicnet, nb_neurons)
        dynamicnet = add_neurons_to_output_layer(dynamicnet, nb_neurons)
        dynamicnet = add_neurons_to_prediction_layer(dynamicnet, nb_neurons)

        ## Project the data on the current basis
        newdata = project_dataset_onto_basis(data, basis)       ## TODO project data piece by piece on the new basis (in the neuralnet_update function)

        ## Define shooting function
        shootingfunc = define_shooting_function(nb_processors, times, solver)

        ## Initialise the optimiser
        optimiser = optimscheme(deep_learning_rate, scheduler)
        optstate = optimiser.init(dynamicnet)

        ## Find the solution to that ODE using PinT
        dynamicnet, loss_ht = neuralnet_update(dynamicnet, newdata, pintscheme, shootingfunc, loss, optstate)

        loss_hts.append(loss_ht)

    return dynamicnet, loss_hts


## A loss that only works for neural ODEs
def node_loss(neuralnet, x, y, loss):
    y_pred = neuralnet.predict(x)
    return loss(y_pred, y)


@jax.jit
def neuralnet_update(neuralnet, dataset, pintscheme, shootingfunc, loss, optimiser, optstate):
    loss_ht = []
    for datapoint, label in dataset:
        fixedpoint = pintscheme(shootingfunc, datapoint)

        loss_val, grad = jax.value_and_grad(node_loss)(neuralnet, fixedpoint, label, loss)

        updates, optstate = optimiser.update(grad, optstate)
        neuralnet = optax.apply_updates(neuralnet, updates)

        loss_ht.append(loss_val)

    return neuralnet, jnp.array(loss_ht)


## Other considerations
# - What is we absolutely don't want to flattent he weights? (e.g. if we want to use a convolutional neural net)