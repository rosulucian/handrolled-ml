import numpy as np
from utils.activations import activations, derivatives


class neural_network():
    def __init__(self, max_iter=200, learning_rate=0.01, layers=[(4, 'relu')], verbose=False):
        self.rand = 0.01
        self.m = 0
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.params = []
        self.layers = layers
        self.verbose = verbose
        self.log_step = max_iter/10

    def init_params(self, X, Y):
        self.m = Y.shape[1]

        # sanity check
        assert(X.shape[1] != 0 and X.shape[1] == self.m)

        self.layers.insert(0, (self.m, 'identity'))  # insert input layer
        self.layers.append((1, 'sigmoid'))  # add output layer; dummy

        for i in range(1, len(self.layers)):
            prev_nodes = self.layers[i-1][0]
            nodes = self.layers[i][0]
            actv = self.layers[i][1]

            layer = {
                'W': np.random.randn(nodes, prev_nodes),
                'b': np.zeros((nodes, 1)),
                'actv': actv
            }

            self.params.append(layer)

    def f_prop(self, X):
        activations = [{'A': X, 'Z': None}]

        for l in range(1, len(self.layers)):
            A_prev = activations[l-1].A
            layer = self.params[l]

            Z = np.dot(layer.W, A_prev) + layer.b
            A = activations[layer.act](Z)
            activations.append({'A': A, 'Z': Z})

        return activations

    def compute_cost(self, AL, Y):
        J = -np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))/self.m
        return J

    def b_prop(self, activations, Y):
        m = self.m
        L = len(self.layers)

        grads = []

        AL = activations[-1][0]
        dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # dAL

        for l in reversed(range(1, L-1)):
            layer = self.params[l]

            Z = activations[l].Z
            A_prev = activations[l-1].A

            dA = dA_prev
            dZ = derivatives[layer.act](dA, Z)
            dW = np.dot(dZ, A_prev.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m

            dA_prev = np.dot(layer.W.T, dZ)  # for next calculation

        return grads

    def update_params(self, grads):
        #

        None

    def fit(self, X, Y):
        self.init_params(X, Y)

        for i in range(self.max_iter):
            activations = self.f_prop(X)

            # compute cost
            AL = activations[-1].A
            cost = self.compute_cost(AL, Y)

            # back prop
            self.b_prop(activations, Y)

            # update params

    def predict(self, X):
        None


# TODO: add checks everywhere
