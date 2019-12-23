from math import exp
import numpy as np


def sigmoid(x):
    return 1. / (1. + exp(-x))


def relu(x):
    return max(0, x)


def sigmoid_grad(x):
    return sigmoid(x) * (1. - sigmoid(x))


def relu_grad(x):
    return x * (x > 0)


v_sigmoid = np.vectorize(sigmoid)
v_relu = np.vectorize(relu)
v_sigmoid_grad = np.vectorize(sigmoid_grad)
v_relu_grad = np.vectorize(relu_grad)
