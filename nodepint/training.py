#%%
## Attributes of an ODE and its trajectory for nodepint

import jax
import jax.numpy as jnp
from equinox import Module
from datasets import Dataset
import optax
from optax import GradientTransformation

from .neuralnets import (DynamicNet, 
                         add_neurons_to_input_layer, add_neurons_to_output_layer, add_neurons_to_prediction_layer,
                         partition_dynamic_net, combine_dynamic_net)
from .pint import define_shooting_function, select_root_finding_function
from .projection import select_projection_scheme
from .data import project_dataset_onto_basis, get_dataset_features


def train_parallel_neural_ode(neural_net:Module, data:Dataset, pint_scheme:str, proj_scheme:str, solver:callable, loss_fn:callable, optim_scheme:GradientTransformation, nb_processors=4, *args, **kwargs):
    ## Steps in the for loop
    # - Sample a vector
    # - Convert the neural net (of class eqx.Module) into a dynamic one (of class DynamicNet)
    # - Project the data on the current basis
    # - Add neurons to the Dynamic NeuralNet's input layer
    # - Find the solution to that ODE using PinT (after setting the `times` attribute) (the `args` indicate extra parameters for the ODE like the start and end times or times vector)
    # - Upscale the feature vector by multiplying by the transposed basis
    # - Evaluate the (augmented) loss function, and backpropagate

    ## WARNINGS:
    # - time needs to be a two-dimensional array (just check this, and add a newaxis if needed)

    ## Get remaiining arguments
    scheduler = kwargs.get("scheduler", optax.constant_schedule(1e-3))
    ## If scheduler is a float, w3rap it in a constant schedule
    if isinstance(scheduler, float):
        scheduler = optax.constant_schedule(scheduler)

    times = kwargs.get("times", jnp.linspace(0, 1, 1000))
    if len(times.shape) == 1:
        times = times[:, jnp.newaxis]

    proj_scheme = select_projection_scheme(proj_scheme)
    print("Projection function name: ", proj_scheme.__name__)

    pint_scheme = select_root_finding_function(pint_scheme)
    print("Time-parallel function name: ", pint_scheme.__name__)

    print("Optimisation scheme is: ", optim_scheme.__name__)

    dynamic_net = DynamicNet(neural_net, pred_size=10)

    max_projection_size = 10 ## TODO make this related to the dataset dimension
    basis = None    ## Initialise the basis
    loss_hts = []    ## Initialise the loss history

    for b in range(max_projection_size):

        ## Sample a vector
        data_feature = list(get_dataset_features(data))[0]
        basis, nb_neurons = proj_scheme(basis, data[data_feature].shape[1:], key=None)

        ## Add neurons to the Dynamic NeuralNet's input, output, and prediction layers
        dynamic_net = add_neurons_to_input_layer(dynamic_net, nb_neurons)
        dynamic_net = add_neurons_to_output_layer(dynamic_net, nb_neurons)
        dynamic_net = add_neurons_to_prediction_layer(dynamic_net, nb_neurons)

        ## Project the data on the current basis
        new_data = project_dataset_onto_basis(data, basis)       ## TODO project data piece by piece on the new basis (in the neuralnet_update function)

        print("New data shape:", new_data[0].shape, "an element", new_data[0][0])

        ## Define shooting function
        shooting_fn = define_shooting_function(nb_processors, times, dynamic_net, solver)

        ## Find the solution to that ODE using PinT
        dynamic_net, loss_ht = neuralnet_update(dynamic_net, new_data, pint_scheme, shooting_fn, nb_processors, loss_fn, optim_scheme, scheduler)

        loss_hts.append(loss_ht)

    return dynamic_net, loss_hts


## A loss that only works for neural ODEs
def node_loss(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors):
    neural_net = combine_dynamic_net(params, static)

    sht_init = jnp.ones((nb_processors+1, x.shape[0]))  ## TODO think of better HOT initialisation
    final_feature = pint_scheme(shooting_fn, z0=x, B0=sht_init)[-1,...]

    print("Final feature shape:", final_feature.shape, "an element", final_feature[0])

    y_pred = neural_net.predict(final_feature)

    return loss_fn(y_pred, y)


# @jax.jit
def neuralnet_update(neural_net, dataset, pint_scheme, shooting_fn, nb_processors, loss_fn, optim_scheme, scheduler):

    ## Partition the dynamic net into static and dynamic parts
    params, static = partition_dynamic_net(neural_net)

    ## Initialise the optimiser
    optimiser = optim_scheme(scheduler)
    optstate = optimiser.init(params)

    print("Dataset:", dataset)

    loss_ht = []
    for data_point, label in zip(*dataset):

        print("Data point:", data_point.shape, "val", data_point, "Label:", label.shape)

        loss_val, grad = jax.value_and_grad(node_loss)(params, static, data_point, label, loss_fn, pint_scheme, shooting_fn, nb_processors)

        updates, optstate = optimiser.update(grad, optstate, params)
        params = optax.apply_updates(params, updates)

        loss_ht.append(loss_val)

    neural_net = combine_dynamic_net(params, static)
    return neural_net, jnp.array(loss_ht)


## Other considerations
# - What is we absolutely don't want to flattent he weights? (e.g. if we want to use a convolutional neural net)