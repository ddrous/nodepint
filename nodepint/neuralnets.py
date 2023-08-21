#%%
## Weights for nodepint package with dynamic input and output layers
#- Use conv nets at the input to dynamize the input size
#- Do not flatten the tensors before sending them into the NeuralODE. You loose inductive bias!


import jax
import equinox as eqx

from .utils import get_key

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

    def __init__(self, neural_net=None, input_size=2, output_size=1, pred_size=1, key=None):
        """ 
        Initialises the input and outputs layers of DynamicNet with input_size (at least 2 to include time) and output_size (at least 1) respectively
        """

        hidden_features = 100 ## TODO get this parameter from the neural_net

        key = get_key(key)
        keys = jax.random.split(key, num=3)

        self.dyn_input_size = input_size
        self.dyn_output_size = output_size
        self.sta_pred_size = pred_size

        self.dyn_input_layer = eqx.nn.Linear(input_size, hidden_features-1, key=keys[0])    ## Remove -1 because time is not placed in the dynamic input layer

        self.sta_hidden_layers = neural_net

        self.dyn_output_layer = eqx.nn.Linear(hidden_features, output_size, key=keys[1])

        if hasattr(neural_net, "prediction_layer"):
            self.dyn_prediction_layer = neural_net.prediction_layer
        else:
            self.dyn_prediction_layer = eqx.nn.Linear(output_size, pred_size, key=keys[2])

    def __call__(self, x, t):
        x = self.dyn_input_layer(x, t)
        x = self.sta_hidden_layers(x)
        return self.dyn_output_layer(x)

    def predict(self, x):
        return self.prediction_layer(x)



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

    old_input_size = neural_net.input_layer.in_features
    old_output_size = neural_net.input_layer.out_features

    new_layer = eqx.nn.Linear(nb_neurons+old_input_size, old_output_size, key=key)

    new_layer.weight = new_layer.weight.at[:, :old_input_size].set(neural_net.input_layer.weight) ## TODO is eqx tree_at faster?: https://docs.kidger.site/equinox/tricks/

    new_layer.bias = neural_net.input_layer.bias

    neural_net.input_layer = new_layer
    neural_net.input_size = new_layer.in_features  ## Just for decoration

    return neural_net





def add_neurons_to_output_layer(neural_net:DynamicNet, nb_neurons:int, key=None):

    key = get_key(key)

    old_input_size = neural_net.output_layer.in_features
    old_output_size = neural_net.output_layer.out_features

    new_layer = eqx.nn.Linear(old_input_size, old_output_size+nb_neurons, key=key)

    new_layer.weight = new_layer.weight.at[:old_output_size, :].set(neural_net.output_layer.weight)

    new_layer.bias = neural_net.output_layer.bias

    neural_net.output_layer = new_layer
    neural_net.output_size = new_layer.out_features

    return neural_net





def add_neurons_to_prediction_layer(neural_net:eqx.Module, basis, key=None):
    key = get_key(key)

    old_input_size = neural_net.prediction_layer.in_features
    old_output_size = neural_net.prediction_layer.out_features

    new_layer = eqx.nn.Linear(basis.shape[1], old_output_size, key=key)

    new_layer.weight = new_layer.weight.at[:, :old_input_size].set(neural_net.input_layer.weight)

    new_layer.bias = neural_net.prediction_layer.bias

    neural_net.prediction_layer = new_layer

    return neural_net




#%%

if __name__ == "__main__":

    input_size = 2
    output_size = 2
    num_hidden_layers = 5

    x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
    x, y = jax.random.normal(x_key, (100, 2)), jax.random.normal(y_key, (100, 2))
    model = DynamicNet(None, input_size, output_size, model_key)

    @eqx.filter_jit
    def loss(model, x, y):
        y_pred = jax.vmap(model)(x, None)
        return jax.numpy.mean((y - y_pred) ** 2)

    print(f"Loss value is: {loss(model, x, y)} \n")

    gradients = eqx.filter_grad(loss)(model, x, y)
    print(f"Gradients PyTree is:\n {gradients}")
