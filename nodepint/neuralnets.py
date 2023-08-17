#%%
## Weights for nodepint package with dynamic input and output layers

import jax
import equinox as eqx

class DynamicMLP(eqx.Module):
    """
    This class is for dynamic multilayer perceptrons. The input and output layers are prone to change in shapes during training. The class ensures that gradients are propagated in a manner that would allow for correct learning.
    """

    input_size: int
    output_size: int

    input_layer: eqx.nn.Linear
    hidden_layers: list
    output_layer: eqx.nn.Linear

    def __init__(self, input_size, output_size, num_hidden_layers, key):

        keys = jax.random.split(key, num=num_hidden_layers+2)
        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = eqx.nn.Linear(input_size, 100, key=keys[0])

        self.hidden_layers = []
        for i in range(num_hidden_layers):
            self.hidden_layers = self.hidden_layers + [eqx.nn.Linear(100, 100, key=keys[i+1]), jax.nn.relu]

        self.output_layer = eqx.nn.Linear(100, output_size, key=keys[-1])

    def reshape_input_layer(self, input_size, key):
        """
        The function reshapes the input layer of a neural network, and makes sure the gradients are propagated to that layer in a manner that would allow faster learning. The key is used to ensure that the same random numbers are generated each time this function is called.

        :param self: Allow an object to refer to itself inside of a method
        :param input_size: Determine the size of the input layer
        :param key: Determine which layer to reshape
        :return: A tensor that is reshaped to the input size
        """
        pass

    def reshape_output_layer(self, output_size, key):
        pass

    def __call__(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


#%%

if __name__ == "__main__":

    input_size = 2
    output_size = 2
    num_hidden_layers = 5

    x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
    x, y = jax.random.normal(x_key, (100, 2)), jax.random.normal(y_key, (100, 2))
    model = DynamicMLP(input_size, output_size, num_hidden_layers, model_key)

    @eqx.filter_jit
    def loss(model, x, y):
        y_pred = jax.vmap(model)(x)
        return jax.numpy.mean((y - y_pred) ** 2)

    print(f"Loss value is: {loss(model, x, y)} \n")

    gradients = eqx.filter_grad(loss)(model, x, y)
    print(f"Gradients PyTree is:\n {gradients}")

