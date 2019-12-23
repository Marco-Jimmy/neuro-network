import model
import numpy as np
import activation_function as af


class optimizer:
    def __init__(self, learning_rate=1, activation_function='sigmoid'):
        optimizer.learning_rate = learning_rate
        if activation_function == 'sigmoid':
            optimizer.activation_function = af.sigmoid
            optimizer.activation_function_grad = af.sigmoid_grad
        elif activation_function == 'relu':
            optimizer.activation_function = af.relu
            optimizer.activation_function_grad = af.relu_grad
