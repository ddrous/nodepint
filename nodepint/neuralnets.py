#%%
## Weights for nodepint package with dynamic input and output layers
#- Use conv nets at the input to dynamize the input size
#- Do not flatten the tensors before sending them into the NeuralODE. You loose inductive bias!


import jax
import equinox as eqx

class DynamicNet(eqx.Module):
    """
    This class is for dynamic multilayer perceptrons. The input and output layers are prone to change in shapes during training.
    """

    input_size: int
    output_size: int

    input_layer: eqx.nn.Linear
    hidden_layers: list
    output_layer: eqx.nn.Linear

    prediction_mapping: eqx.nn.Linear       ## For regression of classification. TODO This layer equally evolves with the output layer

    def __init__(self, neural_net=None, input_size=2, output_size=1, key=None):
        """ 
        Initialises the input and outputs layers of DynamicNet with input_size (at least 2 to include time) and output_size (at least 1) respectively
        """

        num_hidden_layers = 10 ## We don't need to set this up, it should just be whatever the eqx.Module had inside

        keys = jax.random.split(key, num=num_hidden_layers+2)
        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = eqx.nn.Linear(input_size, 100, key=keys[0])

        ## This represents a neural net. TODO use the structure of the eqx.Module as hidden layers
        self.hidden_layers = []
        for i in range(num_hidden_layers):
            self.hidden_layers = self.hidden_layers + [eqx.nn.Linear(100, 100, key=keys[i+1]), jax.nn.relu]

        self.output_layer = eqx.nn.Linear(100, output_size, key=keys[-1])

        self.prediction_mapping = eqx.nn.Linear(output_size, 1, key=keys[-1])

    def __call__(self, x, t):
        x = self.input_layer(x) ## TODO add time to the input layer
        ## TODO The for loop should be changed to whatever the eqx.Module had
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def predict(self, x):
        return self.prediction_mapping(x)



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


def add_neurons_to_input_layer(neural_net:eqx.Module, nb_neurons, key):
    """
    Adds a neuron to the input layer (which must contain at least one neuron before hand)
    - Conserve all hidden existing weights
    - Recreate a new input layer, and copy the old weights that match into their respective positions
    - Reset the properties of the neural net (input size, etc.)
    """
    pass

def add_neurons_to_output_layer(neural_net:eqx.Module, nb_neurons, key):
    pass

def add_neurons_to_prediction_layer(neural_net:eqx.Module, nb_neurons, key):
    pass

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

