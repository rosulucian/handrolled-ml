import numpy as np


activations = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh
}

derivatives = {
    'sigmoid': sigmoid_deriv,
    'relu': relu_deriv,
    'tanh': tanh_deriv
}


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def tanh(Z):
    return np.tanh(Z)


def sigmoid_deriv(dA, Z):
    s = 1/(1+np.exp(-Z))
    deriv = s * (1-s)

    dZ = dA * deriv

    assert (dZ.shape == Z.shape)

    return dZ


def relu_deriv(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def tanh_deriv(dA, Z):
    A = tanh(Z)
    dZ = dA * (1 - np.power(A, 2))

    assert (dZ.shape == Z.shape)

    return dZ
