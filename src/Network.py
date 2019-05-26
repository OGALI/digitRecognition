"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""


import numpy as np
import random
from src import sigmoid


class Network:

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        # The size contains the number of nodes in each layer; input layer is considered a layer
        self.sizes = sizes

        # Initialized biases and weights randomely
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # zip function retunrns a series of tuple with a combo one of each of the list given so that x and y can iterate over
        # creates a matrix of matrices for all the weights where Wij where i is the weight to the second layer node and j is for the first layer node
        # weight matrices represented like that so could do do product of matrices WX + B
        # There is a different weight

    def feedforwad(self, a):
        """Return the output of the network if ``a`` is input."""

        # b and w are acutally matrices to iterate over
        for b, w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def SGD(self, training_data, epoch, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
                gradient descent.  The "training_data" is a list of tuples
                "(x, y)" representing the training inputs and the desired
                outputs.  The other non-optional parameters are
                self-explanatory.  If "test_data" is provided then the
                network will be evaluated against the test data after each
                epoch, and partial progress printed out.  This is useful for
                tracking progress, but slows things down substantially."""

        # If test_data is provided proceed to the following
        if test_data:
            n_test = len(test_data)

        # n is the length of the training data
        n = len(training_data)
        # Redo this same for loop for each epoch
        for j in range(epoch):
            # Shuffles the data
            random.shuffle(training_data)

            # make many small batches after shuffle, not out of bound because range doesnt take the last number
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:

                # For each minibatch, apply single step of the gradient descent and updates weights and biases
                # use 1 mini batch to calculate gradient
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print
                "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print
                "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
                gradient descent using backpropagation to a single mini batch.
                The "mini_batch" is a list of tuples "(x, y)", and "eta"
                is the learning rate."""

        # Initialize the gradients to zero
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            # ***why nibla is calculated for each point
            # calculates the delta to be used required
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # adding the delta to the gradients to get new gradient
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        pass


# Required function to evaluate
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

