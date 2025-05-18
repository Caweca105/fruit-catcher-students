import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        self.input_size = input_size
        # hidden_architecture is a tuple, e.g. (5, 2) for two hidden layers
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        """
        Compute total number of weights (including biases) for this network.
        """
        total = 0
        prev_size = self.input_size
        # Hidden layers
        for layer_size in self.hidden_architecture:
            # weights: prev_size x layer_size, biases: layer_size
            total += prev_size * layer_size + layer_size
            prev_size = layer_size
        # Output layer (single neuron)
        # weights: prev_size x 1, bias: 1
        total += prev_size * 1 + 1
        return total

    def load_weights(self, weights):
        """
        Load a flat list of weights (and biases) into the network.
        """
        w = np.array(weights)
        self.hidden_weights = []
        self.hidden_biases = []

        idx = 0
        prev_size = self.input_size
        # Load hidden layers
        for layer_size in self.hidden_architecture:
            # biases
            b = w[idx: idx + layer_size]
            idx += layer_size
            # weights
            size_w = prev_size * layer_size
            W = w[idx: idx + size_w].reshape(prev_size, layer_size)
            idx += size_w

            self.hidden_biases.append(b)
            self.hidden_weights.append(W)
            prev_size = layer_size

        # Load output layer
        # bias
        self.output_bias = w[idx]
        idx += 1
        # weights
        self.output_weights = w[idx: idx + prev_size]

    def forward(self, x):
        """
        Forward pass: compute network output for input x.
        """
        a = np.array(x)
        # Hidden layers
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, W) + b
            a = self.hidden_activation(z)
        # Output layer
        z_out = np.dot(a, self.output_weights) + self.output_bias
        return self.output_activation(z_out)


def create_network_architecture(input_size):
    """
    Define and return a neural network architecture.

    1) Simple perceptron (single neuron, no hidden layers)
    2) Feedforward network with one hidden layer of 5 neurons
    """
    # Activation functions
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))  # sigmoid
    output_fn = lambda x: 1 if x > 0 else -1    # sign

    # --- Configuration options ---
    # 1) Simple perceptron:
    # Uncomment the next line for a perceptron
    # return NeuralNetwork(input_size, (), hidden_fn, output_fn)

    # 2) Feedforward network with one hidden layer:
    return NeuralNetwork(input_size, (5,), hidden_fn, output_fn)
