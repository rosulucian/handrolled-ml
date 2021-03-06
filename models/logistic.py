import numpy as np
from utils.activations import sigmoid


class logistic:
    def __init__(self, max_iter=100, tol=1e-3, learning_rate=0.01, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.b = 0
        self.verbose = verbose
        self.log_step = max_iter/10

    def propagate(self, X, Y):
        m = X.shape[1]

        # forward propagation
        Z = np.dot(self.W.T, X) + self.b
        A = sigmoid(Z)
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

        for i in range(self.max_iter):
            grads, J = self.propagate(X, Y)

            dw = grads["dw"]
            db = grads["db"]

            # update weights & bias
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if(self.verbose and i % self.log_step == 0):
                print(f'Iteration {i} cost J = {J}')

    def predict(self, X):
        pred = sigmoid(np.dot(self.W.T, X) + self.b)

        pred = (pred > 0.5).astype(float)

        return pred
