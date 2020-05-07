import numpy as np
import math
from .utils.activations import sigmoid


class logistic:
    def __init__(self, max_iter=100, tol=1e-3, learning_rate=0.01):
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.b = 0

    def propagate(self, X, Y):
        m = X.shape[1]

        # forward propagation
        z = np.dot(self.W.T, X) + self.b
        A = sigmoid(z)
        J = -np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))/m

        # back propagation
        dz = (A - Y)  # dL/dz
        dw = 1/m * np.dot(X, dz.T)
        db = 1/m * np.sum(dz)

        grads = {"dw": dw,
                 "db": db}

        return grads, J

    def fit(self, X, Y):
        n = X.shape[0]
        self.W = np.zeros((n, 1))

        for i in range(self.learning_rate):
            grads, J = self.propagate(X, Y)

            dw = grads["dw"]
            db = grads["db"]

            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X, Y):
        pred = sigmoid(np.dot(self.W.T, X) + self.b)

        pred = (pred > 0.5).astype(float)

        return pred
