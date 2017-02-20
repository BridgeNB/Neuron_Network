__author__ = 'zhengyangqiao'

"""
This is a mimic code from Michael Nielson.
I just want to get my hands dirty.
Thanks so much to him that provide such a detailed lesson about
how to construct a neuron network

Back to code
It is a network class that include number of neurons
"""

import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """Return the output of the network of input a"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """This is the neural network using mini-batch stochastic gradient descent.
            The "training_data"is a list of tuples "(x, y)" representing the training inputs and
            the desired outputs. If there is test_data then the network will evaluate test data with
            each epoch, which is for tracking progress but slows down the program"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [ training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation
        to a single mini-batch -> a list of (x, y) tuples
        """
        init_b = [np.zeros(b.shape) for b in self.biases]
        init_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_init_b, delta_init_w = self.backprop(x, y)
            init_b = [nb + delta_b for nb, delta_b in zip(init_b, delta_init_b)]
            init_w = [nw + delta_w for nw, delta_w in zip(init_w, delta_init_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, init_b)]
        self.biases = [b - (eta/ len(mini_batch)) * nb for b, nb in zip(self.biases, init_w)]

def sigmoid(self, z):
    return 1.0/(1.0+np.exp(-z))