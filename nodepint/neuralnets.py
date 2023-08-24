#%%
## Weights for nodepint package with dynamic input and output layers
#- Use conv nets at the input to dynamize the input size
#- Do not flatten the tensors before sending them into the NeuralODE. You loose inductive bias!


import jax
import jax.numpy as jnp
import equinox as eqx

from .utils import get_key
# from utils import get_key

class DynamicNet(eqx.Module):
    """
    This class is for dynamic multilayer perceptrons. The input and output layers are prone to change in shapes during training.
    """

    dyn_input_size: int
    dyn_output_size: int
    sta_pred_size: int

    dyn_input_layer: eqx.nn.Linear
    sta_hidden_layers: eqx.Module
    dyn_output_layer: eqx.nn.Linear

    dyn_prediction_layer: eqx.nn.Linear

    def __init__(self, neural_net=None, input_size=1, output_size=0, pred_size=1, key=None):
        """ 
        Initialises the input and outputs layers of DynamicNet with input_size (at least 2 to include time) and output_size (at least 1) respectively
        """

        # hidden_features = 100 ## TODO get this parameter from the neural_net

        key = get_key(key)
        keys = jax.random.split(key, num=3)

        self.dyn_input_size = input_size
        self.dyn_output_size = output_size
        self.sta_pred_size = pred_size

        if neural_net is not None:
            self.sta_hidden_layers = neural_net
        else:
            self.sta_hidden_layers = eqx.nn.MLP(in_size=100, out_size=100, width_size=250, depth=5, activation=jax.nn.relu, key=model_key)

        ## Print all attributes of the neural net
        print("Hidden net attributes are: ", self.sta_hidden_layers.__dict__.keys())

        ## TODO what follows is highly brittle (only works for MLPs)
        in_hidden_features = self.sta_hidden_layers.layers[0].in_features
        out_hidden_features = self.sta_hidden_layers.layers[-1].out_features

        self.dyn_input_layer = eqx.nn.Linear(input_size, in_hidden_features, key=keys[0])    ## TODO what if the user actually implements an input layer, or a conv net...?

        self.dyn_output_layer = eqx.nn.Linear(out_hidden_features, output_size, key=keys[1])

        # if hasattr(neural_net, "prediction_layer"):     ## TODO make sure the user never sets a prediction layer
        #     self.dyn_prediction_layer = neural_net.prediction_layer
        # else:
        print("output and pred size:", output_size, pred_size)
        # pred_in_size = output_size if output_size>0 else 1
        # lim = 1/(pred_in_size+1)
        self.dyn_prediction_layer = eqx.nn.Linear(output_size+1, pred_size, key=keys[2])
        glorot_weights = glorot_uniform(shape=(pred_size, output_size), key=keys[2])
        self.dyn_prediction_layer = eqx.tree_at(lambda l: l.weight, self.dyn_prediction_layer, glorot_weights)

    def __call__(self, x, t):
        # print("Values and shapes of t and x:", t.shape, x.shape)

        tx = jnp.concatenate([t, x], axis=-1)
        # print("tx size", tx.shape, self.dyn_input_size)
        # print("dyn input layer", self.dyn_input_layer)
        y = self.dyn_input_layer(tx)
        y = self.sta_hidden_layers(y)
        y = self.dyn_output_layer(y)
        # print("Output layer:", self.dyn_output_layer)
        # print("y output shape", y.shape)
        # print("dyn output layer", self.dyn_output_layer)
        return y

    def predict(self, x):
        ## Check shapes
        print("X before prediction shape", x.shape)
        print("dyn prediction layer", self.dyn_prediction_layer)
        return self.dyn_prediction_layer(x)


def glorot_uniform(shape, key):
    shape_1 = shape[1] if len(shape)>1 else 0
    return jax.random.uniform(key, shape) * jnp.sqrt(6 / (shape[0] + shape_1))


def reshape_input_layer(neural_net:eqx.Module, input_size, key):
    """
    The function reshapes the input layer of a neural network, and makes sure the gradients are propagated to that layer in a manner that would allow faster learning. The key is used to ensure that the same random numbers are generated each time this function is called.

    :param self: Allow an object to refer to itself inside of a method
    :param input_size: Determine the size of the input layer
    :param key: Determine which layer to reshape
    :return: A tensor that is reshaped to the input size
    """
    pass

def reshape_output_layer(neural_net:eqx.Module, output_size, key):
    pass

## TIP make sure all the side effects are centred in this function
def add_neurons_to_input_layer(neural_net:DynamicNet, nb_neurons:int, key=None):
    """
    Adds a neuron to the input layer (which must contain at least one neuron before hand)
    - Conserve all hidden existing weights
    - Recreate a new input layer, and copy the old weights that match into their respective positions
    - Reset the properties of the neural net (input size, etc.)
    """
    key = get_key(key)

    old_output_size, old_input_size = neural_net.dyn_input_layer.weight.shape

    # new_layer_weight = jax.random.normal(key, shape=(old_output_size, nb_neurons+old_input_size))
    new_layer_weight = glorot_uniform(shape=(old_output_size, nb_neurons+old_input_size), key=key)
    new_layer_weight = new_layer_weight.at[:, :old_input_size].set(neural_net.dyn_input_layer.weight)

    where = lambda nn: nn.dyn_input_layer.weight
    return eqx.tree_at(where, neural_net, new_layer_weight)





def add_neurons_to_output_layer(neural_net:DynamicNet, nb_neurons:int, key=None):

    key = get_key(key)

    old_output_size, old_input_size = neural_net.dyn_output_layer.weight.shape

    # new_layer_weight = jax.random.normal(key, shape=(old_output_size+nb_neurons, old_input_size))
    new_layer_weight = glorot_uniform(shape=(old_output_size+nb_neurons, old_input_size), key=key)
    new_layer_weight = new_layer_weight.at[:old_output_size, :].set(neural_net.dyn_output_layer.weight)

    where = lambda nn: nn.dyn_output_layer.weight
    neural_net = eqx.tree_at(where, neural_net, new_layer_weight)

    # new_layer_bias = jax.random.normal(key, shape=(old_output_size+nb_neurons,))
    new_layer_bias = glorot_uniform(shape=(old_output_size+nb_neurons, ), key=key)
    new_layer_bias = new_layer_bias.at[:old_output_size].set(neural_net.dyn_output_layer.bias)

    where = lambda nn: nn.dyn_output_layer.bias
    return eqx.tree_at(where, neural_net, new_layer_bias)


def partition_dynamic_net(neural_net:DynamicNet):
    params, static = eqx.partition(neural_net, eqx.is_array)
    return params, static

def combine_dynamic_net(params, static):
    return eqx.combine(params, static)

def add_neurons_to_prediction_layer(neural_net:DynamicNet, nb_neurons:int, key=None):
    key = get_key(key)

    old_output_size, old_input_size = neural_net.dyn_prediction_layer.weight.shape

    # old_input_size = neural_net.dyn_prediction_layer.in_features
    ## old_input_size = 0 if old_input_size=="scalar" else old_input_size
    # old_output_size = neural_net.dyn_prediction_layer.out_features

    # new_layer_weight = jax.random.normal(key, shape=(old_output_size, old_input_size+nb_neurons))
    new_layer_weight = glorot_uniform(shape=(old_output_size, old_input_size+nb_neurons), key=key)
    ## Print stuff before assign
    # print("Dyn layer", neural_net.dyn_prediction_layer.weight.shape, "new layer", new_layer_weight.shape)
    print("old input size", old_input_size, "new input size", old_input_size+nb_neurons)

    new_layer_weight = new_layer_weight.at[:, :old_input_size].set(neural_net.dyn_prediction_layer.weight)

    where = lambda nn: nn.dyn_prediction_layer.weight
    return eqx.tree_at(where, neural_net, new_layer_weight)




#%%

if __name__ == "__main__":

    input_size = 3
    output_size = 2
    num_hidden_layers = 5

    x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
    x, y = jax.random.normal(x_key, (100, 2)), jax.random.normal(y_key, (100, 2))
    t = jnp.linspace(0, 1, 1000)[:, jnp.newaxis][-1]

    hidden_mlp = eqx.nn.MLP(in_size=100, out_size=100, width_size=250, depth=num_hidden_layers, activation=jax.nn.relu, key=model_key)
    # print(f"Hidden MLP is: {hidden_mlp} \n")

    model = DynamicNet(hidden_mlp, input_size, output_size, pred_size=2, key=model_key)
    # print(f"Dynamic model is: {model} \n")

    @eqx.filter_jit
    def loss(model, t, x, y):
        y_pred = jax.vmap(model, in_axes=(None, 0))(t, x)
        return jax.numpy.mean((y - y_pred) ** 2)

    print(f"Loss value is: {loss(model, t, x, y)} \n")

    gradients = eqx.filter_grad(loss)(model, t, x, y)
    print(f"Gradients PyTree is:\n {gradients}")
