import numpy as np
from utils.activations import sigmoid


class toy_shallow_nn():
    def __init__(self, max_iter=200, learning_rate=0.01, hidden_nodes=4, verbose=False):
        self.rand = 0.01
        self.m = 0
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.hidden_nodes = hidden_nodes
        self.verbose = verbose
        self.log_step = max_iter/10

    def init_params(self, X, Y):
        # sanity check
        assert(X.shape[1] == Y.shape[1] and X.shape[1] != 0)

        self.W1 = self.rand * np.random.randn(self.hidden_nodes, X.shape[0])
        self.b1 = np.zeros((self.hidden_nodes, 1))
        self.W2 = self.rand * np.random.randn(Y.shape[0], self.hidden_nodes)
        self.b2 = np.zeros((Y.shape[0], 1))

        self.m = Y.shape[1]

    def f_prop(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)

        return A1, A2

    def compute_cost(self, A2, Y):
        J = -np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2))/self.m

        return J

    def b_prop(self, A1, A2, X, Y):
        m = self.m

        dZ2 = A2-Y
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m

        # tanh`(Z1) = 1 - A1^2 = 1 - tanh(Z1)^2
        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2):
        lr = self.learning_rate

        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def fit(self, X, Y):
        self.init_params(X, Y)

        for i in range(self.max_iter):
            A1, A2 = self.f_prop(X)

            J = self.compute_cost(A2, Y)

            dW1, db1, dW2, db2 = self.b_prop(A1, A2, X, Y)

            self.update_params(dW1, db1, dW2, db2)

            if(self.verbose and i % self.log_step == 0):
                print(f'Iteration {i} cost J = {J}')

    def predict(self, X):
        _, A2 = self.f_prop(X)
        predictions = (A2 > 0.5).astype(float)

        return predictions
