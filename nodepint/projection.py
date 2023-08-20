#%%
## Sampling strategies for the nodepint package

## Sampling can be done sin three ways :
# - woth one radom vector at a time
# - with one random vector at a time to construct a basis
# - with one backpropagated vector at a time
# - with a reduced informative basis determined before hand by the user (using PCA on the ground thruth for instance)
# - with a reduced basis learned simultanuously with the Node (using SINDy for instance)

## The sampling strategy is determined by the user, and is passed as an argument to the `train_parallel_neural_ode` function


import jax
def random_sampling(basis:jax.Array, key:jax.random.PRNGKey=None):

    if key is None:
        key = jax.random.PRNGKey(0)

    while True:
        key, subkey = jax.random.split(key)
        yield jax.random.normal(subkey, shape=basis.shape[1])

def learned_sampling(basis:jax.Array, key:jax.random.PRNGKey=None):
    pass

def select_projection_scheme(name:str):
    if name=="random":
        scheme = random_sampling
    elif name=="learned":
        scheme = learned_sampling

    return scheme
