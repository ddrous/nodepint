#%%
## Attributes of an ODE and its trajectory for nodepint

import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module
from datasets import Dataset
import optax
from optax import GradientTransformation
from functools import partial

from .neuralnets import (DynamicNet, 
                         add_neurons_to_input_layer, add_neurons_to_output_layer, add_neurons_to_prediction_layer,
                         partition_dynamic_net, combine_dynamic_net)
from .pint import select_root_finding_function, shooting_function
from .projection import select_projection_scheme
from .data import get_dataset_features
from .utils import get_new_keys



def train_parallel_neural_ode(neural_net:Module, data:Dataset, pint_scheme:str, proj_scheme:str, integrator:callable, loss_fn:callable, optim_scheme:GradientTransformation, nb_processors:int, nb_epochs:int, batch_size:int, scheduler:float, times: tuple, repeat_projection:int, nb_vectors:int, key=None):
    ## Steps in the for loop
    # - Sample a vector
    # - Convert the neural net (of class eqx.Module) into a dynamic one (of class DynamicNet)
    # - Project the data on the current basis
    # - Add neurons to the Dynamic NeuralNet's input layer
    # - Find the solution to that ODE using PinT (after setting the `times` attribute) (the `args` indicate extra parameters for the ODE like the start and end times or times vector)
    # - Upscale the feature vector by multiplying by the transposed basis
    # - Evaluate the (augmented) loss function, and backpropagate

    if isinstance(scheduler, float):
        scheduler = optax.constant_schedule(scheduler)

    if not isinstance(times, tuple):
        times = tuple(times.flatten())

    if isinstance(proj_scheme, str):
        proj_scheme = select_projection_scheme(proj_scheme)
    print("Projection function name: ", proj_scheme.__name__)

    ## TODO use many projections per-data points
    if isinstance(pint_scheme, str):
        pint_scheme = select_root_finding_function(pint_scheme)
    # print("Time-parallel function name: ", pint_scheme.__name__)

    print("Optimisation scheme is: ", optim_scheme.__name__)
    print("Integrator is: ", integrator.__name__)

    ## Setup features for later
    all_features = get_dataset_features(data)
    data_feature, label_feature = all_features[0], all_features[-1]

    model_key, proj_key = get_new_keys(key, 2)

    pred_size = int(np.prod(data[0][label_feature].shape))
    dynamic_net = DynamicNet(neural_net, pred_size=pred_size, key=model_key)

    print("Dynamic net construction, done !")

    basis = None    ## Initialise the basis
    loss_hts = []    ## Initialise the loss history

    for p in range(repeat_projection):

        vec_size = jnp.prod(jnp.asarray(data[0][data_feature].shape[:]))

        ## Sample a vector
        basis, nb_neurons = proj_scheme(old_basis=basis, orig_vec_size=vec_size, nb_new_vecs=nb_vectors, key=proj_key)

        print("\nBasis constructed, with shape:", basis.shape)

        ## Add neurons to the Dynamic NeuralNet's input, output, and prediction layers
        keys = get_new_keys(model_key, 3)
        dynamic_net = add_neurons_to_input_layer(dynamic_net, nb_neurons, key=keys[0])
        dynamic_net = add_neurons_to_output_layer(dynamic_net, nb_neurons, key=keys[1])
        dynamic_net = add_neurons_to_prediction_layer(dynamic_net, nb_neurons, key=keys[2])

        print("Adding neurons to dynamic net's layers, done !")

        ## Find the solution to that ODE using PinT and backpropagate
        dynamic_net, loss_ht = train_dynamic_net(dynamic_net, data, basis, pint_scheme, shooting_function, nb_processors, times, integrator, loss_fn, optim_scheme, scheduler, nb_epochs, batch_size)

        loss_hts.append(loss_ht)

    return dynamic_net, basis, shooting_function, loss_hts


def train_dynamic_net(neural_net, dataset, basis, pint_scheme, shooting_fn, nb_processors, times, integrator, loss_fn, optim_scheme, scheduler, nb_epochs, batch_size):

    ## Partition the dynamic net into static and dynamic parts
    params, static = partition_dynamic_net(neural_net)

    ## Initialise the optimiser
    optimiser = optim_scheme(scheduler)
    optstate = optimiser.init(params)

    features = get_dataset_features(dataset)


    # batch_size = 1

    loss_ht = []
    for epoch in range(nb_epochs):

        loss_eph = 0
        nb_batches = 0
        # dataset = project_dataset_onto_basis(dataset, basis)
        # for x, y in zip(*dataset):
        # print("batch size:", batch_size)

        for batch in dataset.iter(batch_size=batch_size):
            x, y = batch[features[0]], batch[features[1]]
            x = x.reshape((x.shape[0], -1)) @ basis

            params, optstate, loss_val = train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, optimiser, optstate)

            loss_eph += jnp.sum(loss_val)
            nb_batches += 1

        loss_eph /= nb_batches
        print("Epoch: %-5d      Loss: %.6f" % (epoch, loss_eph))

        loss_ht.append(loss_eph)

    neural_net = combine_dynamic_net(params, static)
    return neural_net, jnp.array(loss_ht)



@partial(jax.jit, static_argnames=("static", "loss_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator", "optimiser"))
def train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, optimiser, optstate):

    loss_val, grad = jax.value_and_grad(node_loss)(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator)

    updates, optstate = optimiser.update(grad, optstate, params)
    params = optax.apply_updates(params, updates)

    return params, optstate, loss_val


# from nodepint.pint import newton_root_finder, direct_root_finder
# batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None), out_axes=0)
# batched_pint_scheme = jax.vmap(direct_root_finder, in_axes=(None, None, 0, None, None, None, None, None, None, None), out_axes=0)

# batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)



## A loss that only works for neural ODEs
def node_loss(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator):
    neural_net = combine_dynamic_net(params, static)

    sht_init = jnp.ones((nb_processors+1, x.shape[1])).flatten()  ## TODO think of better HOT initialisation. Parareal ?

    # batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None), out_axes=0)
    batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None, None, None, None, None), out_axes=0)
    batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)

    # batched_pint_scheme = pint_scheme
    # batched_model_pred = neural_net.predict

    # print("x type:", type(x))
    # print("Types of other params:", type(shooting_fn), type(sht_init), x.shape, type(nb_processors), type(times), type(neural_net), type(integrator), type(1.), type(1e-6), type(3))

    ## Jax print debug trace
    # jax.debug.breakpoint()

    # final_feature = batched_pint_scheme(shooting_fn, sht_init, x, nb_processors, times, neural_net, integrator, 1., 1e-6, 3)[:, -x.shape[1]:]
    final_feature = batched_pint_scheme(shooting_fn, sht_init, x, nb_processors, times, params, static, integrator, 1., 1e-6, 3)[:, -x.shape[1]:]

    y_pred = batched_model_pred(final_feature)

    # return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0))(y_pred, y))

    return jnp.mean(loss_fn(y_pred, y))  ## TODO loss_fn should be vmapped by design


## Other considerations
# - What is we absolutely don't want to flattent he weights? (e.g. if we want to use a convolutional neural net)












def test_dynamic_net(neural_net, data, basis, pint_scheme, shooting_fn, nb_processors, times, integrator, acc_fn, batch_size):

    ## Partition the dynamic net into static and dynamic parts
    params, static = partition_dynamic_net(neural_net)

    features = get_dataset_features(data)

    nb_batches = 0
    total_acc = 0.
    for batch in data.iter(batch_size=batch_size):
        x, y = batch[features[0]], batch[features[1]]
        x = x.reshape((x.shape[0], -1)) @ basis

        acc_val = test_step(params, static, x, y, acc_fn, pint_scheme, shooting_fn, nb_processors, times, integrator)

        nb_batches += 1
        total_acc += acc_val

    return total_acc/nb_batches


@partial(jax.jit, static_argnames=("static", "acc_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator"))
def test_step(params, static, x, y, acc_fn, pint_scheme, shooting_fn, nb_processors, times, integrator):

    neural_net = combine_dynamic_net(params, static)

    sht_init = jnp.ones((nb_processors+1, x.shape[1])).flatten()  ## TODO think of better HOT initialisation. Parareal ?

    batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None), out_axes=0)

    batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)

    final_feature = batched_pint_scheme(shooting_fn, sht_init, x, nb_processors, times, neural_net, integrator)[:, -x.shape[1]:]

    y_pred = batched_model_pred(final_feature)

    return acc_fn(y_pred, y)
