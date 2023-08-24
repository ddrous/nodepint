#%%
## Sampling strategies for the nodepint package

## Sampling can be done sin three ways :
# - woth one radom vector at a time
# - with one random vector at a time to construct a basis
# - with one backpropagated vector at a time
# - with a reduced informative basis determined before hand by the user (using PCA on the ground thruth for instance)
# - with a reduced basis learned simultanuously with the Node (using SINDy for instance)

## The sampling strategy is determined by the user, and is passed as an argument to the `train_parallel_neural_ode` function


from typing import Collection
import jax
import jax.numpy as jnp

from .utils import get_key

def random_sampling(old_basis:jax.Array=None, shape:Collection=(None, 1), key:jax.random.PRNGKey=None):

    key = get_key(key)

    if old_basis is None:
        if [s for s in shape if s is None]:
            raise ValueError("If no basis is provided, the shape of the random vector must be specified")
        # print("new basis: ", jax.random.normal(key, shape=(jnp.prod(jnp.asarray(shape)), 1)))

        ret = jax.random.normal(key, shape=(jnp.prod(jnp.asarray(shape)), 1)), 1
    else:
        new_basis = jax.random.normal(key, shape=shape)
        new_basis = new_basis.at[..., :old_basis.shape[-1]].set(old_basis)

        ret = new_basis, new_basis.shape[-1]-old_basis.shape[-1]

    #     print("new basis: ", new_basis)
    # print("bases shapes: ", old_basis, ret[0].shape)
    return ret

def learned_sampling(basis:jax.Array, key:jax.random.PRNGKey=None):
    pass

def select_projection_scheme(name:str="random"):
    if name=="random":
        scheme = random_sampling
    elif name=="learned":
        scheme = learned_sampling
    else:
        raise ValueError("Unknown projection scheme")
    return scheme
