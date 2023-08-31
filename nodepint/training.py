#%%
## Attributes of an ODE and its trajectory for nodepint

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
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
from .data import project_dataset_onto_basis, get_dataset_features, extract_all_data


def train_parallel_neural_ode(neural_net:Module, data:Dataset, pint_scheme:str, proj_scheme:str, integrator:callable, loss_fn:callable, optim_scheme:GradientTransformation, nb_processors=4, nb_epochs=10, *args, **kwargs):
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

    times = kwargs.get("times", tuple(np.linspace(0, 1, 100).flatten()))
    # if len(times.shape) == 1:
    #     times = times[:, jnp.newaxis]

    proj_scheme = select_projection_scheme(proj_scheme)
    print("Projection function name: ", proj_scheme.__name__)

    ## TODO use many projections per-data points

    pint_scheme = select_root_finding_function(pint_scheme)
    print("Time-parallel function name: ", pint_scheme.__name__)

    print("Optimisation scheme is: ", optim_scheme.__name__)
    print("Integrator is: ", integrator.__name__)

    dynamic_net = DynamicNet(neural_net, pred_size=10)

    print("Dynamic net construction, done !")

    max_projection_size = 4 ## TODO make this related to the dataset dimension
    basis = None    ## Initialise the basis
    loss_hts = []    ## Initialise the loss history

    for b in range(max_projection_size):

        ## Get the size of the data
        # data_feature = list(get_dataset_features(data))[0]
        data_feature = "image"
        vec_size = jnp.prod(jnp.asarray(data[0][data_feature].shape[:]))
        ## Sample a vector
        basis, nb_neurons = proj_scheme(old_basis=basis, orig_vec_size=vec_size, nb_new_vecs=2, key=None)

        print("\nBasis constructed, with shape:", basis.shape)

        ## Add neurons to the Dynamic NeuralNet's input, output, and prediction layers
        dynamic_net = add_neurons_to_input_layer(dynamic_net, nb_neurons)
        dynamic_net = add_neurons_to_output_layer(dynamic_net, nb_neurons)
        dynamic_net = add_neurons_to_prediction_layer(dynamic_net, nb_neurons)

        ## Project the data on the current basis
        # new_data = project_dataset_onto_basis(data, basis)       ## TODO project data piece by piece on the new basis (in the neuralnet_update function)

        print("Adding neurons to dynamic net's layers, done !")

        # print("New data shape:", new_data[0].shape, "an element", new_data[0][0])

        ## Define shooting function
        # shooting_fn = define_shooting_function(nb_processors, times, dynamic_net, integrator)

        ## Find the solution to that ODE using PinT
        dynamic_net, loss_ht = neuralnet_update(dynamic_net, data, basis, pint_scheme, shooting_function, nb_processors, times, integrator, loss_fn, optim_scheme, scheduler, nb_epochs)

        loss_hts.append(loss_ht)

    return dynamic_net, basis, loss_hts


# @partial(eqx.filter_jit, static_argnames=("pint_scheme", "shooting_fn", "nb_processors"))
def neuralnet_update(neural_net, dataset, basis, pint_scheme, shooting_fn, nb_processors, times, integrator, loss_fn, optim_scheme, scheduler, nb_epochs):

    ## Partition the dynamic net into static and dynamic parts
    params, static = partition_dynamic_net(neural_net)

    ## Initialise the optimiser
    optimiser = optim_scheme(scheduler)
    optstate = optimiser.init(params)

    # print("Dataset:", dataset)

    loss_ht = []
    for epoch in range(nb_epochs):

        loss_eph = 0

        # dataset = project_dataset_onto_basis(dataset, basis)
        # for x, y in zip(*dataset):

        for batch in dataset.iter(batch_size=500):      ## TODO! the following 3 lines if vmap is used
            x, y = batch["image"], batch["label"]
            x = x.reshape((x.shape[0], -1)) @ basis

            # print("Data point:", data_point.shape, "val", data_point, "Label:", label.shape)

            # loss_val, grad = jax.value_and_grad(node_loss)(params, static, data_point, label, loss_fn, pint_scheme, shooting_fn, nb_processors)

            # updates, optstate = optimiser.update(grad, optstate, params)
            # params = optax.apply_updates(params, updates)

            # loss_eph += loss_val

            # print("Shapes just before training step:", x.shape, y.shape)

            # p_leaves = jax.tree_util.tree_leaves(params)
            # o_leaves = jax.tree_util.tree_leaves(optstate)
            # print(f"params has {len(p_leaves)} with shape {p_leaves[0].shape}")
            # print(f"opstate has {len(o_leaves)} with shape {o_leaves[0].shape}")

            params, optstate, loss_val = train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, optimiser, optstate)

            # print('Loss val', loss_val)

            loss_eph += jnp.sum(loss_val)

        print(f" Epoch: {epoch} \t Loss: {loss_eph}")

        loss_ht.append(loss_eph)

    neural_net = combine_dynamic_net(params, static)
    return neural_net, jnp.array(loss_ht)



# @partial(jax.vmap, in_axes=(None, None, 0, 0, None, None, None, None, None, None, None, None), out_axes=0)
@partial(jax.jit, static_argnames=("static", "loss_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator", "optimiser"))
def train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, optimiser, optstate):


    loss_val, grad = jax.value_and_grad(node_loss)(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator)

    # batched_loss_fn = jax.vmap(jax.value_and_grad(node_loss), in_axes=(None, None, 0, 0, None, None, None, None, None, None), out_axes=0)

    # loss_val, grad = batched_loss_fn(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator)

    # g_leaves = jax.tree_util.tree_leaves(grad)
    # print(f"GRRRRAD grads has {len(g_leaves)} with shape {g_leaves[0].shape}")

    updates, optstate = optimiser.update(grad, optstate, params)
    params = optax.apply_updates(params, updates)

    return params, optstate, loss_val


## A loss that only works for neural ODEs
# @partial(jax.vmap, in_axes=(None, None, 0, 0, None, None, None, None, None, None), out_axes=0)
def node_loss(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator):
    neural_net = combine_dynamic_net(params, static)

    sht_init = jnp.ones((nb_processors+1, x.shape[1])).flatten()  ## TODO think of better HOT initialisation

    # print("x shape:", x.shape, "an element", x[0], "res", pint_scheme(shooting_fn, z0=x, B0=sht_init).shape)

    # final_feature = pint_scheme(shooting_fn, B0=sht_init, z0=x, nb_splits=nb_processors, times=times, rhs=neural_net, integrator=integrator)[-x.shape[0]:]

    # y_pred = neural_net.predict(final_feature)


    batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None, None, None, None), out_axes=0)
    batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)

    # final_feature = batched_pint_scheme(shooting_fn, B0=sht_init, z0=x, nb_splits=nb_processors, times=times, rhs=neural_net, integrator=integrator, learning_rate=1., tol=1e-6, maxiter=3)[:, -x.shape[1]:]

    final_feature = batched_pint_scheme(shooting_fn, sht_init, x, nb_processors, times, neural_net, integrator, 1., 1e-6, 3)[:, -x.shape[1]:]

    y_pred = batched_model_pred(final_feature)


    # print("y_pred shape:", y_pred.shape, "y shape:", y.shape, "feauture", final_feature.shape)
    # return loss_fn(y_pred, y)

    return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0))(y_pred, y))


## Other considerations
# - What is we absolutely don't want to flattent he weights? (e.g. if we want to use a convolutional neural net)
