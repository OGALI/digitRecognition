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


    def feedforwad(self, a):
        """Return the output of the network if ``a`` is input."""

        # b and w are acutally matrices to iterate over
        for b, w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def SGD(self, training_data, epoch, mini_batch, eta, test_data=None):
        pass


