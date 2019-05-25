import numpy as np
from src import sigmoid


class Network:

    def __init__(self, sizes):
        '''Constructor for the network class'''

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
        '''Returns the output of the network is a is the input'''

        # b and w are acutally matrices to iterate over
        for b, w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def SGD(self, training_data, epoch, mini_batch, eta, test_data=None):
        pass

