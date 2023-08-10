#%%
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt
import seaborn as sns

#%%
def mlp(params, inputs):
    for w, b in params[:-1]:
        outputs = jnp.dot(inputs, w) + b
        inputs = jnp.tanh(outputs)
    final_w, final_b = params[-1]
    outputs = jnp.dot(inputs, final_w) + final_b
    return outputs


def resnet(params, inputs, depth=2):    ## ! A single MLP does well here (TRY using different MLP params at each depth)
    for i in range(depth):
        inputs = mlp(params, inputs) + inputs
    return inputs


def resnet_squared_loss(params, inputs, targets):
    preds = resnet(params, inputs)
    return jnp.mean(jnp.sum((preds - targets)**2, axis=-1))

def init_random_params(layer_sizes, key=PRNGKey(0)):
    keys = jax.random.split(key, len(layer_sizes))
    return [[jax.random.normal(k, (m, n)), jax.random.normal(k, (n,))]
            for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]

@jax.jit
def resnet_update(params, inputs, targets, learning_rate=0.01):
    grads = jax.grad(resnet_squared_loss)(params, inputs, targets)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(params, grads)]


inputs = jnp.reshape(jnp.linspace(-2., 2., 10), (10, 1))
targets = inputs**3 + 0.1 * inputs

layer_sizes = [1, 20, 1]

resnet_params = init_random_params(layer_sizes)
for i in range(1000):
    resnet_params = resnet_update(resnet_params, inputs, targets)

fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.gca()
ax.plot(inputs, targets, "o", label='data')

fine_inputs = jnp.reshape(jnp.linspace(-3., 3., 100), (100, 1))
predictions = resnet(resnet_params, fine_inputs)

ax.plot(fine_inputs, predictions, label='predictions')
ax.set(xlabel='inputs', ylabel='outputs')
ax.legend()

# %%

def nn_dynamics(state, time, params):
    """Compute the derivative of the state at a given time."""
    state_time = jnp.hstack([state, time])
    return mlp(params, state_time)


# @jax.tree_util.Partial(jax.vmap, in_axes=(None,None,0,None,None))
def neural_ode(func, params, y0):
    t0, tf = 0., 1.
    start_finish = jnp.array([t0, tf])
    init_state, final_state = odeint(func, y0, start_finish, params)
    return final_state

batched_neural_ode = jax.vmap(neural_ode, in_axes=(None, None, 0), out_axes=0)

ode_layer_sizes = [2, 20, 1]

def neural_ode_loss(params, inputs, targets):
    preds = batched_neural_ode(nn_dynamics, params, inputs)
    return jnp.mean(jnp.sum((preds - targets)**2, axis=-1))

@jax.jit
def neural_ode_update(params, inputs, targets, learning_rate=0.01):
    grads = jax.grad(neural_ode_loss)(params, inputs, targets)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(params, grads)]

ode_params = init_random_params(ode_layer_sizes)

for i in range(1000):
    ode_params = neural_ode_update(ode_params, inputs, targets)

fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.gca()
ax.plot(inputs, targets, "o", label='data')

node_predictions = batched_neural_ode(nn_dynamics, ode_params, fine_inputs)
ax.plot(fine_inputs, predictions, label='resnet')
ax.plot(fine_inputs, node_predictions, label='neural ode')
ax.set(xlabel='inputs', ylabel='outputs')
ax.legend()


# %%
## ! Neural ODEs in time. Activation trajectories

fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.gca()


@jax.jit
def neural_ode_times(params, inputs, times):
    def func(state, time, params):
        return mlp(params, jnp.hstack([state, time]))
    return odeint(func, inputs, times, params)

times = jnp.linspace(0.0, 1.0, 200)

for i, input in enumerate(fine_inputs):
    node_predictions = neural_ode_times(ode_params, input, times)
    ax.plot(node_predictions, times)

ax.set(xlabel='inputs', ylabel='time/depth')
ax.legend()



# %%

## FINDINGS
# Continue the tutorial here: http://implicit-layers-tutorial.org/neural_odes/

# 1. Neural ODEs are a generalization of ResNets
# 2. Resnets are not good if you use the same MLP at multiple depths. The tutorial has a bug. Their resnet always returns the last layers output; without compounding the depths.
