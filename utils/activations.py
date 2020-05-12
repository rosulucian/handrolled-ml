import numpy as np


activations = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    return np.tanh(z)
