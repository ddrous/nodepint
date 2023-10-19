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
from .pint import select_root_finding_function, fixed_point_ad, shooting_function_serial, shooting_function_parallel
from .projection import select_projection_scheme
from .data import get_dataset_features
from .utils import get_new_keys, increase_timespan
from .integrators import euler_integrator



def train_parallel_neural_ode(neural_net:Module, data:Dataset, pint_scheme:str, proj_scheme:str, integrator:callable, loss_fn:callable, optim_scheme:GradientTransformation, nb_processors:int, nb_epochs:int, batch_size:int, scheduler:float, times: tuple, fixed_point_args:tuple, repeat_projection:int, nb_vectors:int, force_serial:bool=False, key=None):
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

    ## TODO use many projections per-data points. The Pint scheme can be any root finder, or parareal
    if isinstance(pint_scheme, str):
        pint_scheme = select_root_finding_function(pint_scheme)
    print("Root-finding strategy name: ", pint_scheme.__name__)

    ## To force time intervals to be computed in a for look
    if force_serial == "disguised_parallel":    ## TODO Make this better
        shooting_function = shooting_function_serial
    else:
        shooting_function = shooting_function_parallel

    print("Optimisation scheme is: ", optim_scheme.__name__)
    print("Integrator is: ", integrator.__name__)

    if force_serial == False:
        print("Shooting function is: ", shooting_function.__name__)
        print("Total number of time intervals: ", nb_processors)
        print("Number of JAX devices available:", len(jax.devices()))
        print("Number of time intervals per device:", nb_processors//len(jax.devices()))
    elif force_serial == "disguised_parallel":
        print("Shooting function is: ", shooting_function.__name__)
        print("Total number of time intervals: ", nb_processors)
    else:
        print("WARNING: you are running the integrator serialy")

    ## Setup features for later
    all_features = get_dataset_features(data)
    data_feature, label_feature = all_features[0], all_features[-1]

    model_key, proj_key = get_new_keys(key, 2)

    pred_size = int(np.prod(data[0][label_feature].shape))
    dynamic_net = DynamicNet(neural_net, pred_size=pred_size, key=model_key)

    print("Dynamic net construction, done !")

    basis = None    ## Initialise the basis
    loss_hts = []    ## Initialise the loss history
    errors_hts = []
    nb_iters_hts = []
    for p in range(repeat_projection):

        vec_size = jnp.prod(jnp.asarray(data[0][data_feature].shape[:]))

        ## Sample a vector
        if p>0 and proj_scheme.__name__=="identity_sampling":
            print("Cannot perform basis augmentation with identity sampling")
            break
        else:
            basis, nb_neurons = proj_scheme(old_basis=basis, orig_vec_size=vec_size, nb_new_vecs=nb_vectors, key=proj_key)

        print("\nBasis constructed, with shape:", basis.shape)

        ## Add neurons to the Dynamic NeuralNet's input, output, and prediction layers
        keys = get_new_keys(model_key, 3)
        dynamic_net = add_neurons_to_input_layer(dynamic_net, nb_neurons, key=keys[0])
        dynamic_net = add_neurons_to_output_layer(dynamic_net, nb_neurons, key=keys[1])
        dynamic_net = add_neurons_to_prediction_layer(dynamic_net, nb_neurons, key=keys[2])

        print("Adding neurons to dynamic net's layers, done !")

        ## Find the solution to that ODE using PinT and backpropagate
        dynamic_net, loss_ht, errors, nb_iters = train_dynamic_net(dynamic_net, data, basis, pint_scheme, shooting_function, nb_processors, times, integrator, fixed_point_args, loss_fn, optim_scheme, scheduler, nb_epochs, batch_size, force_serial)

        loss_hts.append(loss_ht)
        errors_hts.append(errors)
        nb_iters_hts.append(nb_iters)

    return dynamic_net, basis, shooting_function, loss_hts, errors_hts, nb_iters_hts


def train_dynamic_net(neural_net, dataset, basis, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, loss_fn, optim_scheme, scheduler, nb_epochs, batch_size, force_serial):

    ## Partition the dynamic net into static and dynamic parts
    params, static = partition_dynamic_net(neural_net)

    ## Initialise the optimiser
    optimiser = optim_scheme(scheduler)
    optstate = optimiser.init(params)

    features = get_dataset_features(dataset)

    print_every = nb_epochs//10 if nb_epochs>10 else 1
    # batch_size = 1

    loss_ht = []
    errors = []
    nb_iters = []
    for epoch in range(nb_epochs):

        loss_eph = 0
        nb_batches = 0
        # dataset = project_dataset_onto_basis(dataset, basis)
        # for x, y in zip(*dataset):
        # print("batch size:", batch_size)

        for batch in dataset.iter(batch_size=batch_size):
            x, y = batch[features[0]], batch[features[1]]
            x = x.reshape((x.shape[0], -1)) @ basis

            params, optstate, loss_val, aux_data = train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, optimiser, optstate, force_serial)

            errors.append(aux_data[0])      ## TODO return something meaningfull per epoch
            nb_iters.append(aux_data[1])

            loss_eph += jnp.sum(loss_val)
            nb_batches += 1

        loss_eph /= nb_batches

        if epoch<3 or epoch%print_every==0:
            print("Epoch: %-5d      Loss: %.6f" % (epoch, loss_eph))

        loss_ht.append(loss_eph)

    neural_net = combine_dynamic_net(params, static)
    return neural_net, jnp.array(loss_ht), errors, nb_iters



@partial(jax.jit, static_argnames=("static", "loss_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator", "optimiser", "fixed_point_args", "force_serial"))
def train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, optimiser, optstate, force_serial):

    (loss_val, aux_data), grad = jax.value_and_grad(node_loss, has_aux=True)(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, force_serial)

    updates, optstate = optimiser.update(grad, optstate, params)
    params = optax.apply_updates(params, updates)

    return params, optstate, loss_val, aux_data


# from nodepint.pint import newton_root_finder, direct_root_finder
# batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None), out_axes=0)
# batched_pint_scheme = jax.vmap(direct_root_finder, in_axes=(None, None, 0, None, None, None, None, None, None, None), out_axes=0)

# batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)



## A loss that only works for neural ODEs
def node_loss(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, force_serial):
    neural_net = combine_dynamic_net(params, static)


    if force_serial == False:
        ## For time-paralle computing

        ## increase timespan for euler not to diverge
        factor = np.ceil((times[1]-times[0])/(nb_processors * times[3])).astype(int)
        t_init = increase_timespan(jnp.linspace(times[0], times[1], nb_processors+1), factor)

        print("Factor for PinT initialisation is:", factor)     ## Much needed side effect !
        feat0 = jax.vmap(euler_integrator, in_axes=(None, None, 0, None, None))(params, static, x, t_init, np.inf)[:, ::factor, :]

        lr, tol, max_iter = fixed_point_args
        batched_pint = jax.vmap(fixed_point_ad, in_axes=(None, 0, 0, None, None, None, None, None, None, None, None, None), out_axes=0)
        features, errors, nb_iters = batched_pint(shooting_fn, feat0, x, nb_processors, times, params, static, integrator, pint_scheme, lr, tol, max_iter)

    else:
        ## For serial computing
        batched_odeint = jax.vmap(integrator, in_axes=(None, None, 0, None, None), out_axes=0)
        features = batched_odeint(params, static, x, jnp.linspace(*times[:3]), times[3])
        errors, nb_iters = None, None


    batched_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)
    y_pred = batched_pred(features[:, -1, :])

    # return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0))(y_pred, y))
    return jnp.mean(loss_fn(y_pred, y)), (errors, nb_iters)  ## TODO loss_fn should be vmapped by design


## Other considerations
# - What is we absolutely don't want to flattent he weights? (e.g. if we want to use a convolutional neural net)












def test_dynamic_net(neural_net, data, basis, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, acc_fn, batch_size):

    ## Partition the dynamic net into static and dynamic parts
    params, static = partition_dynamic_net(neural_net)

    features = get_dataset_features(data)

    nb_batches = 0
    total_acc = 0.
    for batch in data.iter(batch_size=batch_size):
        x, y = batch[features[0]], batch[features[1]]
        x = x.reshape((x.shape[0], -1)) @ basis

        acc_val = test_step(params, static, x, y, acc_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args)

        nb_batches += 1
        total_acc += acc_val

    return total_acc/nb_batches


@partial(jax.jit, static_argnames=("static", "acc_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator", "fixed_point_args"))
def test_step(params, static, x, y, acc_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args):

    neural_net = combine_dynamic_net(params, static)

    sht_init = jnp.ones((nb_processors+1, x.shape[1])).flatten()  ## TODO think of better HOT initialisation. Parareal ?

    batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None, None, None, None, None), out_axes=0)

    batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)

    final_feature = batched_pint_scheme(shooting_fn, sht_init, x, nb_processors, times, params, static, integrator, 1., 1e-6, 3)[:, -x.shape[1]:]

    y_pred = batched_model_pred(final_feature)

    return acc_fn(y_pred, y)
