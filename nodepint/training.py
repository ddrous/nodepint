#%%
## Attributes of an ODE and its trajectory for nodepint

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from equinox import Module
# from datasets import Dataset
from torch.utils.data import Dataset
import optax
from optax import GradientTransformation
from functools import partial

from .neuralnets import DynamicNet
from .pint import select_root_finding_function, fixed_point_ad, shooting_function_serial, shooting_function_parallel, direct_root_finder_aug, parareal
from .sampling import select_projection_scheme
from .data import make_dataloader_torch
from .utils import get_new_keys, increase_timespan
from .integrators import euler_integrator

# import jax.profiler
# jax.profiler.start_server(9999)

def train_project_neural_ode(neural_nets:tuple, data:Dataset, pint_scheme:str, samp_scheme:str, integrator:callable,integrator_args:tuple, loss_fn:callable, optim_scheme:GradientTransformation, nb_processors:int, nb_epochs:int, batch_size:int, scheduler:float, times: tuple, fixed_point_args:tuple, repeat_projection:int, nb_vectors:int, force_serial:bool=False, key=None):
    ## Steps in the for loop
    # - Sample a vector
    # - Convert the neural net (of class eqx.Module) into a dynamic one (of class DynamicNet)
    # - Project the data on the current basis
    # - Add neurons to the Dynamic NeuralNet's input layer
    # - Find the solution to that ODE using PinT (after setting the `times` attribute) (the `args` indicate extra parameters for the ODE like the start and end times or times vector)
    # - Upscale the feature vector by multiplying by the transposed basis
    # - Evaluate the (augmented) loss function, and backpropagate

    ### Train while projecting the neural ODE onto a smaller basis


    if isinstance(scheduler, float):
        scheduler = optax.constant_schedule(scheduler)

    if not isinstance(times, tuple):
        times = tuple(times.flatten())

    encoder, processor, decoder = neural_nets

    if isinstance(samp_scheme, str):
        samp_scheme = select_projection_scheme(samp_scheme)
    print("Sampling function name: ", samp_scheme.__name__)


    if encoder is None:
        neural_encoding = False
        if samp_scheme.__name__ == "neural_sampling":
            print("ERROR: you need an encoder to use the neural sampling scheme. Define an encoder or change the sampling scheme")
    else:
        neural_encoding = True
        if samp_scheme.__name__ != "neural_sampling":
            print("WARNING: you can only use an encoder with a neural sampling scheme. Enforcing the neural sampling scheme")
            samp_scheme = "neural"


    ## TODO use many projections per-data points. The PinT scheme can be any root finder, or parareal
    if isinstance(pint_scheme, str):
        pint_scheme = select_root_finding_function(pint_scheme)
    
    if force_serial == False:
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
    # all_features = get_dataset_features(data)
    # data_feature, label_feature = all_features[0], all_features[-1]

    model_key, proj_key = get_new_keys(key, 2)

    # ## If the first layer of the processor is linear, then we are working with MLPs
    # mlp_setting = isinstance(neural_nets[1][0], eqx.nn.Linear)

    if not neural_encoding:
        # pred_size = int(np.prod(data[0][label_feature].shape))
        pred_size = int(np.prod(data[0][1].shape))

        dynamic_encoder = DynamicNet(None, pred_size=pred_size, key=model_key)
        dynamic_processor = DynamicNet(processor, pred_size=pred_size, key=model_key)
        dynamic_decoder = DynamicNet(decoder, pred_size=pred_size, key=model_key)

        neural_nets = (dynamic_encoder, dynamic_processor, dynamic_decoder)

        print("Dynamic net construction, done !")



    basis_wrap = neural_nets[0]      ## The wrapper for the basis is the encoder
    loss_hts = []                    ## Initialise the loss history
    errors_hts = []
    nb_iters_hts = []
    for p in range(repeat_projection):

        if not neural_encoding:

            vec_size = jnp.prod(jnp.asarray(data[0][0].shape[:]))

            ## Sample a vector
            if p>0 and samp_scheme.__name__=="identity_sampling":
                print("Cannot perform basis augmentation with identity sampling")
                break
            else:
                basis_wrap, nb_neurons = samp_scheme(old_basis=basis_wrap, orig_vec_size=vec_size, nb_new_vecs=nb_vectors, key=proj_key)

            neural_nets = basis_wrap, neural_nets[1], neural_nets[2]


            # print("\nBasis constructed, with shape:", basis_wrap.shape)

            ## Add neurons to the Dynamic NeuralNet's input, output, and prediction layers
            # keys = get_new_keys(model_key, 3)
            # neural_nets = add_neurons_to_input_layer(neural_nets, nb_neurons, key=keys[0])
            # neural_nets = add_neurons_to_output_layer(neural_nets, nb_neurons, key=keys[1])
            # neural_nets = add_neurons_to_prediction_layer(neural_nets, nb_neurons, key=keys[2])

            ## TODO Search the entire processor and decoder for layers sensible to shape changes, namely linear layers

            print("Adding neurons to dynamic net's layers, done !")

        ## Find the solution to that ODE using PinT and backpropagate
        neural_nets, loss_ht, errors, nb_iters = train_neural_ode(neural_nets, data, pint_scheme, shooting_function, nb_processors, times, integrator, integrator_args, fixed_point_args, loss_fn, optim_scheme, scheduler, nb_epochs, batch_size, force_serial)

        loss_hts.append(loss_ht)
        errors_hts.append(errors)
        nb_iters_hts.append(nb_iters)

    # jax.profiler.print_summary()

    return neural_nets, shooting_function, loss_hts, errors_hts, nb_iters_hts


# def train_neural_ode(neural_nets, dataset, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, loss_fn, optim_scheme, scheduler, nb_epochs, batch_size, force_serial):

#     ## Partition the networks net into static and dynamic parts
#     params, static = eqx.partition(neural_nets, eqx.is_array)

#     ## Initialise the optimiser
#     optimiser = optim_scheme(scheduler)
#     optstate = optimiser.init(params)

#     # features = get_dataset_features(dataset)
#     # print_every = nb_epochs//10 if nb_epochs>10 else 1
#     if nb_epochs > 100:
#         print_every = nb_epochs//100
#     elif nb_epochs > 10:
#         print_every = nb_epochs//10
#     else:
#         print_every = 1

#     dataloader = make_dataloader_torch(dataset, batch_size=batch_size)


#     loss_ht = []
#     errors = []
#     nb_iters = []
#     for epoch in range(1, nb_epochs+1):

#         loss_eph = 0
#         nb_batches = 0

#         for batch in dataloader:
#             x, y = jnp.array(batch[0]), jnp.array(batch[1])

#             # x = x.reshape((x.shape[0], -1)) @ basis

#             params, optstate, loss_val, aux_data = train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args, optimiser, optstate, force_serial)

#             # errors.append(aux_data[0])      ## TODO return something meaningfull per epoch
#             # nb_iters.append(aux_data[1])

#             errors.append(jnp.max(aux_data[0], axis=0))      ## TODO return something meaningfull per epoch
#             nb_iters.append(jnp.max(aux_data[1]))

#             loss_eph += jnp.sum(loss_val)
#             nb_batches += 1

#             # if nb_batches >= 5:
#             #     break

#         loss_eph /= nb_batches

#         if epoch<=3 or epoch%print_every==0:
#             print("Epoch: %-5d      Loss: %.6f" % (epoch, loss_eph))

#         loss_ht.append(loss_eph)

#     return eqx.combine(params, static), jnp.array(loss_ht), errors, nb_iters


def train_neural_ode(neural_nets, dataset, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args, loss_fn, optim_scheme, scheduler, nb_epochs, batch_size, force_serial):

    ## Partition the networks net into static and dynamic parts
    params, static = eqx.partition(neural_nets, eqx.is_array)

    ## Initialise the optimiser
    optimiser = optim_scheme(scheduler)
    optstate = optimiser.init(params)

    # features = get_dataset_features(dataset)

    if nb_epochs > 100:
        print_every = nb_epochs//100
    elif nb_epochs > 10:
        print_every = nb_epochs//10
    else:
        print_every = 1

    nb_batches = int(np.ceil(len(dataset)/batch_size))
    max_pint_iters = fixed_point_args[-1]

    losses = jnp.zeros((nb_epochs, nb_batches))
    errors = jnp.zeros((nb_epochs, nb_batches, max_pint_iters))
    nb_iters = jnp.zeros((nb_epochs, nb_batches))

    dataloader = make_dataloader_torch(dataset, batch_size=batch_size)

    for epoch in range(nb_epochs):
        batch_idx = 0

        for batch in dataloader:
            x, y = jnp.array(batch[0]), jnp.array(batch[1])
           
            params, optstate, loss_val, aux_data = train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args, optimiser, optstate, force_serial)

            losses, errors, nb_iters = per_batch_callback(losses, errors, nb_iters, loss_val, aux_data[0], aux_data[1], epoch, batch_idx)

            batch_idx += 1

            # if batch_idx >= 5:
            #     break

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print("Epoch: %-5d      Loss: %.6f" % (epoch, jnp.mean(losses[epoch])))
            # print("Epoch: %-5d      Loss: %.6f" % (epoch, jnp.sum(losses[epoch])))

    assert batch_idx == nb_batches, "ERROR: The number of batches is not correct"

    return eqx.combine(params, static), jnp.mean(losses, axis=-1), errors, nb_iters
    # return eqx.combine(params, static), jnp.sum(losses, axis=-1)/batch_idx, errors, nb_iters



@jax.jit
def per_batch_callback(losses, errors, nb_iters, loss_val, error_val, iter_val, epoch_idx, batch_idx):
    """A callback function to be called at the end of each batch each epoch."""

    losses = losses.at[epoch_idx, batch_idx].set(loss_val)

    errors = errors.at[epoch_idx, batch_idx, :].set(jnp.max(error_val, axis=0))
    nb_iters = nb_iters.at[epoch_idx, batch_idx].set(jnp.max(iter_val))

    return losses, errors, nb_iters




@partial(jax.jit, static_argnames=("static", "loss_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator", "integrator_args", "optimiser", "fixed_point_args", "force_serial"))
def train_step(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args, optimiser, optstate, force_serial):

    (loss_val, aux_data), grad = jax.value_and_grad(node_loss_fn, has_aux=True)(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args, force_serial)

    ## Remember to freeze encoder gradients if proj scheme is neural TODO

    updates, optstate = optimiser.update(grad, optstate, params)
    params = optax.apply_updates(params, updates)

    return params, optstate, loss_val, aux_data


# from nodepint.pint import newton_root_finder, direct_root_finder
# batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None), out_axes=0)
# batched_pint_scheme = jax.vmap(direct_root_finder, in_axes=(None, None, 0, None, None, None, None, None, None, None), out_axes=0)

# batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)



## A loss that only works for neural ODEs
def node_loss_fn(params, static, x, y, loss_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args, force_serial):

    neural_nets = eqx.combine(params, static)
    encoder, processor, decoder = neural_nets

    ## Project the data on the encoder basis
    batched_encoder = jax.vmap(encoder, in_axes=(0), out_axes=0)
    x_enc = batched_encoder(x)

    ## Only the processor is partioned for the neural ODE
    params_proc, static_proc = eqx.partition(processor, eqx.is_array)

    rtol, atol, hmax, max_steps, max_steps_rev, kind = integrator_args
    lr, tol, max_iter = fixed_point_args

    if force_serial == False:
        ## For time-paralle computing

        ## increase timespan for euler not to diverge
        factor = np.ceil((times[1]-times[0])/(nb_processors * hmax)).astype(int) if np.isfinite(hmax) else 10
        t_init = increase_timespan(jnp.linspace(times[0], times[1], nb_processors+1), factor)

        print("Factor for PinT initialisation is:", factor)     ## Much needed side effect !
        x_proc_coarse = jax.vmap(euler_integrator, in_axes=(None, None, 0, None, None, None, None, None, None, None))(params_proc, static_proc, x_enc, t_init, rtol, atol, hmax, max_steps, max_steps_rev, kind)[:, ::factor, ...]

        # batched_processor = jax.vmap(fixed_point_ad, in_axes=(None, 0, 0, None, None, None, None, None, None, None, None, None), out_axes=0)
        # x_proc_fine, errors, nb_iters = batched_processor(shooting_fn, x_proc_coarse, x_enc, nb_processors, times, params_proc, static_proc, integrator, pint_scheme, lr, tol, max_iter)

        batched_processor = jax.vmap(direct_root_finder_aug, in_axes=(None, 0, 0, None, None, None, None, None, None, None, None), out_axes=0)
        x_proc_fine, errors, nb_iters = batched_processor(shooting_fn, x_proc_coarse, x_enc, nb_processors, times, params_proc, static_proc, integrator, lr, tol, max_iter)

    else:

        ## For serial computing
        # t_eval = jnp.array([times[0], times[1]])
        # t_eval = jnp.array([times[1]])

        if integrator.__name__ in ("euler_integrator", "rk4_integrator"):    ## Fixed steps sizes, users must know this !
            t_eval = jnp.linspace(times[0], times[1], times[2])
        if y.ndim == x.ndim+1:
            t_eval = jnp.linspace(times[0], times[1], y.shape[1])         ## Fully observed adaptive times-stepping dynamical system !
        else:
            # t_eval = jnp.array([times[1]])                                  ## We only care for the final time step !
            t_eval = jnp.linspace(times[0], times[1], 2)

        batched_processor = jax.vmap(integrator, in_axes=(None, None, 0, None, None, None, None, None, None, None), out_axes=0)
        # x_proc_fine = batched_odeint(params_proc, static_proc, x_enc, jnp.linspace(*times[:3]), times[3])
        # x_proc_fine = batched_processor(params_proc, static_proc, x_enc, t_eval, rtol=rtol, atol=atol, hmax=hmax, mxstep=max_steps, max_steps_rev=max_steps_rev, kind=kind)
        x_proc_fine = batched_processor(params_proc, static_proc, x_enc, t_eval, rtol, atol, hmax, max_steps, max_steps_rev, kind)

        errors, nb_iters = None, None

        ## Testing abscence of a integrator !!
        # batched_odeint = jax.vmap(processor, in_axes=(0, None), out_axes=0)
        # x_proc_fine = batched_odeint(x_enc, jnp.zeros((1,)+x_enc.shape[2:]))[:, None, ...]

        max_iter = fixed_point_args[-1]
        errors, nb_iters = jnp.inf*jnp.ones((max_iter,)), 0

    batched_decoder = jax.vmap(decoder, in_axes=(0), out_axes=0)
    y_pred = batched_decoder(x_proc_fine[:, -1, ...])       ## TODO!! the loss function should take care of this !!!!

    # y = jax.nn.one_hot(y, y_pred.shape[-1])

    # return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0))(y_pred, y))
    return jnp.mean(loss_fn(y_pred, y)), (errors, nb_iters)  ## TODO loss_fn should be vmapped by design


## Other considerations
# - What is we absolutely don't want to flattent he weights? (e.g. if we want to use a convolutional neural net)












def test_neural_ode(neural_nets, data, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args, acc_fn, batch_size):

    ## Partition the dynamic net into static and dynamic parts
    params, static = eqx.partition(neural_nets, eqx.is_array)

    dataloader = make_dataloader_torch(data, batch_size=batch_size)

    nb_examples = 0
    total_acc = 0.
    for batch in dataloader:
        x, y = jnp.array(batch[0]), jnp.array(batch[1])

        acc_val = test_step(params, static, x, y, acc_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args)

        nb_examples += x.shape[0]
        total_acc += acc_val

    return total_acc/nb_examples


# @partial(jax.jit, static_argnames=("static", "acc_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator", "fixed_point_args"))
# def test_step(params, static, x, y, acc_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, fixed_point_args):

#     neural_net = combine_dynamic_net(params, static)

#     sht_init = jnp.ones((nb_processors+1, x.shape[1])).flatten()  ## TODO think of better HOT initialisation. Parareal ?

#     batched_pint_scheme = jax.vmap(pint_scheme, in_axes=(None, None, 0, None, None, None, None, None, None, None, None), out_axes=0)

#     batched_model_pred = jax.vmap(neural_net.predict, in_axes=(0), out_axes=0)

#     final_feature = batched_pint_scheme(shooting_fn, sht_init, x, nb_processors, times, params, static, integrator, 1., 1e-6, 3)[:, -x.shape[1]:]

#     y_pred = batched_model_pred(final_feature)

#     return acc_fn(y_pred, y)


@partial(jax.jit, static_argnames=("static", "acc_fn", "pint_scheme", "shooting_fn", "nb_processors", "times", "integrator", "integrator_args", "fixed_point_args"))
def test_step(params, static, x, y, acc_fn, pint_scheme, shooting_fn, nb_processors, times, integrator, integrator_args, fixed_point_args):

    neural_nets = eqx.combine(params, static)
    encoder, processor, decoder = neural_nets

    batched_encoder = jax.vmap(encoder, in_axes=(0), out_axes=0)
    x_enc = batched_encoder(x)

    params_proc, static_proc = eqx.partition(processor, eqx.is_array)

    if integrator.__name__ in ("euler_integrator", "rk4_integrator"):
        t_eval = jnp.linspace(times[0], times[1], times[2])
    if y.ndim == x.ndim+1:
        t_eval = jnp.linspace(times[0], times[1], y.shape[1]) 
    else:
        t_eval = jnp.linspace(times[0], times[1], 2)

    rtol, atol, hmax, max_steps, max_steps_rev, kind = integrator_args

    batched_odeint = jax.vmap(integrator, in_axes=(None, None, 0, None, None, None, None, None, None, None), out_axes=0)
    x_proc_fine = batched_odeint(params_proc, static_proc, x_enc, t_eval, rtol, atol, hmax, max_steps, max_steps_rev, kind)

    batched_decoder = jax.vmap(decoder, in_axes=(0), out_axes=0)
    y_pred = batched_decoder(x_proc_fine[:, -1, ...])

    return acc_fn(y_pred, y)
