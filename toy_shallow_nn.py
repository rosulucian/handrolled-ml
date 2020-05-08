import numpy as np


class toy_shallow_nn():
    def __init__(self, max_iter=100, learning_rate=0.05, hidden_nodes=4):
        self.rand = 0.01
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.hidden_nodes = hidden_nodes
        self.m = 0

    def init_params(self, X, Y):
        # sanity check
        assert(X.shape[1] == Y.shape[1] and X.shape[1] != 0)

        self.W1 = self.rand * np.random.randn(self.hidden_nodes, X.shape[0])
        self.b1 = np.zeros((self.hidden_nodes, 1))
        self.W2 = self.rand * np.random.randn(Y.shape[0], self.hidden_nodes)
        self.b2 = np.zeros((sizes[2], 1))

        self.m = Y.shape[1]

    def f_prop(self, X, Y):
        Z1 = np.dot(self.W1.T, X) + self.b1
        A1 = np.tanh(Z1)

        Z2 = np.dot(self.W2.T, X1) + self.b2
        A2 = sigmoid(Z2)

        J = -np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2))/m

        return A1, A2, J

    def b_prop(self, A1, A2, X, Y):
        dZ2 = A2-Y
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m

        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # tanh`(Z1) = 1 - A1^2
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2):
        lr = self.learning_rate
        W1 -= lr * dW1
        b1 -= lr * db1

        W2 -= lr * dW2
        b2 -= lr * db2

    def fit(self, X, Y):
        init_params(X, Y)

        for i range(self.max_iter):
            # forward prop
            # compute cost
            A1, A2, J = f_prop(X, Y)

            # back prop
            dW1, db1, dW2, db2 = b_prop(A1, A2, X, Y)

            # update params
            update_params(dW1, db1, dW2, db2)

    def predict(self, X, Y):
