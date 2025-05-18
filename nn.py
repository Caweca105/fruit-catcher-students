import numpy as np

class NeuralNetwork:
    """
    Flexible feedforward neural network (including perceptron) for the Fruit Catcher AI.

    Attributes:
    - input_size: dimensionality of the state vector (should be 10)
    - hidden_architecture: tuple of ints, each the number of neurons in a hidden layer
    - hidden_activation: activation function for hidden layers (vectorized)
    - output_activation: activation function for output layer (scalar)

    Weights are stored flat and can be loaded via `load_weights`. Predictions use `forward`.
    """
    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Placeholders for weights/biases; to be set via load_weights()
        self.hidden_weights = []
        self.hidden_biases = []
        self.output_weights = None
        self.output_bias = None

    def compute_num_weights(self):
        """
        Compute the total number of parameters (weights + biases) in the network.
        """
        total = 0
        prev_size = self.input_size
        # Hidden layers: each layer has prev_size * layer_size weights + layer_size biases
        for layer_size in self.hidden_architecture:
            total += prev_size * layer_size + layer_size
            prev_size = layer_size
        # Output layer (one neuron): prev_size weights + 1 bias
        total += prev_size * 1 + 1
        return total

    def load_weights(self, flat_weights):
        """
        Unpack a flat list or array of weights into layer-wise matrices and bias vectors.
        """
        w = np.array(flat_weights, dtype=float)
        expected = self.compute_num_weights()
        if w.size != expected:
            raise ValueError(f"Expected {expected} weights, got {w.size}")

        idx = 0
        prev_size = self.input_size
        self.hidden_weights = []
        self.hidden_biases = []

        # Hidden layers
        for layer_size in self.hidden_architecture:
            # biases for this layer
            b = w[idx : idx + layer_size]
            idx += layer_size
            # weights for this layer
            size_w = prev_size * layer_size
            W = w[idx : idx + size_w].reshape(prev_size, layer_size)
            idx += size_w

            self.hidden_biases.append(b)
            self.hidden_weights.append(W)
            prev_size = layer_size

        # Output layer
        # bias (scalar)
        self.output_bias = float(w[idx])
        idx += 1
        # weights from last hidden (or input) to output
        self.output_weights = w[idx : idx + prev_size]

    def forward(self, x):
        """
        Perform a forward pass and return the network's scalar output (action).

        x: array-like of length input_size
        returns: output_activation applied to the final linear combination
        """
        a = np.array(x, dtype=float)
        # Propagate through hidden layers
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, W) + b
            a = self.hidden_activation(z)
        # Final output neuron
        z_out = np.dot(a, self.output_weights) + self.output_bias
        return self.output_activation(z_out)


def create_network_architecture(input_size):
    """
    Define the neural network architecture for evolution.

    Returns a NeuralNetwork instance. Two configurations are provided:
      1) Single perceptron (no hidden layers).
      2) One hidden layer of 5 neurons.

    Activate the desired one by uncommenting its `return` line.
    """
    # Activation functions
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))  # sigmoid for hidden layers
    output_fn = lambda x: 1 if x > 0 else -1    # signum for action decision

    # --- Configuration options ---
    # 1) Simple perceptron (no hidden layer)
    # return NeuralNetwork(input_size, (), hidden_fn, output_fn)

    # 2) Feedforward network with one hidden layer of 5 neurons
    return NeuralNetwork(input_size, (5,), hidden_fn, output_fn)
