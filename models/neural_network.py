import numpy as np
import utils.activations as act_fct
import utils.regularization as regl
import functools


class neural_network():
    def __init__(self, max_iter=200, learning_rate=0.01, layers=[(4, 'relu'), (1, 'sigmoid')], regularization=None, lbd=0.7, verbose=False):
        self.rand = 0.05
        self.reg = regularization
        self.lbd = lbd
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
        np.random.seed(1)

        # insert input layer; dummy
        self.layers.insert(0, (X.shape[0], 'identity'))

        for i in range(1, len(self.layers)):
            prev_nodes = self.layers[i-1][0]
            nodes = self.layers[i][0]
            actv = self.layers[i][1]

            layer = {
                'W': np.random.randn(nodes, prev_nodes) * self.rand,
                'b': np.zeros((nodes, 1)),
                'actv': actv
            }

            self.params.append(layer)

        # insert dummy node 0
        self.params.insert(0, {'W': None, 'b': None, 'actv': None})

    def f_prop(self, X):
        activations = [{'A': X, 'Z': None}]

        for l in range(1, len(self.layers)):
            A_prev = activations[l-1].get('A')
            layer = self.params[l]

            Z = np.dot(layer.get('W'), A_prev) + layer.get('b')
            A = act_fct.forward[layer.get('actv')](Z)

            activations.append({'A': A, 'Z': Z})

        return activations

    def compute_cost(self, AL, Y):
        J = -np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))/self.m
        np.squeeze(J)

        if self.reg:
            param_list = list(map(lambda x: x['W'], self.params[1:]))
            J += regl.reg_cost[self.reg](param_list, self.lbd, self.m)

        return J

    def b_prop(self, activations, Y):
        m = self.m
        L = len(self.layers)

        grads = []

        AL = activations[-1].get('A')
        dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # dAL

        for l in reversed(range(1, L)):  # skip 0th layer; dummy layer
            layer = self.params[l]

            W = layer.get('W')
            Z = activations[l].get('Z')
            A_prev = activations[l-1].get('A')

            dA = dA_prev
            dZ = act_fct.backward[layer.get('actv')](dA, Z)
            dW = np.dot(dZ, A_prev.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m

            if self.reg:
                dW -= regl.reg_grads[self.reg](W, self.lbd, self.m)

            grads.insert(0, {'dW': dW, 'db': db})

            dA_prev = np.dot(W.T, dZ)  # for next calculation

            assert (dA_prev.shape == A_prev.shape)
            assert (dW.shape == layer.get('W').shape)
            assert (db.shape == layer.get('b').shape)

        grads.insert(0, {'dW': None, 'db': None})  # dummy grad for input

        return grads

    def update_params(self, grads):
        for l in range(1, len(self.params)):
            # layer = self.params[l]

            assert(self.params[l]['W'].shape == grads[l].get('dW').shape)
            assert(self.params[l]['b'].shape == grads[l].get('db').shape)

            self.params[l]['W'] -= self.learning_rate * grads[l].get('dW')
            self.params[l]['b'] -= self.learning_rate * grads[l].get('db')

    def fit(self, X, Y):
        self.init_params(X, Y)

        for i in range(self.max_iter):
            activations = self.f_prop(X)

            # compute cost
            AL = activations[-1].get('A')
            J = self.compute_cost(AL, Y)

            # back prop
            grads = self.b_prop(activations, Y)

            # update params
            self.update_params(grads)

            if(self.verbose and i % 100 == 0):
                print(f'Iteration {i} cost J = {J}')

    def predict(self, X):
        A = self.f_prop(X)[-1].get('A')
        predictions = (A > 0.5).astype(float)

        return predictions

# TODO: add checks everywhere
