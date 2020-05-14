import numpy as np
import functools


def L2_cost(param_list, lambd, m):
    summed = sum(map(lambda x: np.sum(np.square(x)), param_list))
    return summed * lambd / (m*2)


reg_cost = {
    'L2': L2_cost,
}


def L2_grads(W, lambd, m):
    return lambd * W / m


reg_grads = {
    'L2': L2_grads,
}
