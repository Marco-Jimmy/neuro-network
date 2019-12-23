from activation_function import *
import numpy as np


class Network:
    def __init__(self, input_array=None, activation_function='sigmoid'):

        self.input_vector = []
        self.output_vector = []
        self.weight = []
        self.bias = []
        self.result = []
        self.value = []
        self.error = []
        self.weight_gradient = []
        self.bias_gradient = []
        self.learning_rate = 0.1

        self.input_len = input_array[0]
        self.depth = len(input_array)

        for layer in range(self.depth):
            if layer != self.depth - 1:
                self.weight.append(np.random.random((input_array[layer + 1], input_array[layer])))
                self.weight_gradient.append(np.random.random((input_array[layer + 1], input_array[layer])))
            if layer != 0:
                self.bias.append(np.random.random((input_array[layer])))
                self.bias_gradient.append(np.random.random((input_array[layer])))
            self.error.append(np.random.random((input_array[layer])))
            self.result.append(np.random.random((input_array[layer])))
            self.value.append(np.random.random((input_array[layer])))

        if activation_function == 'sigmoid':
            self.activation_function = v_sigmoid
            self.activation_function_grad = v_sigmoid_grad
        elif activation_function == 'relu':
            self.activation_function = relu
            self.activation_function_grad = v_relu_grad

    def input(self, input_vector=None):
        if input_vector is None:
            input_vector = []
        assert len(input_vector) == self.input_len
        self.input_vector = np.array(input_vector)
        self.result[0] = self.input_vector

    def feed_forward(self, input_vector):
        self.input(input_vector)
        for layer in range(1, self.depth):
            self.value = np.dot(self.weight[layer - 1], self.result[layer - 1]) + self.bias[layer - 1]
            self.result[layer] = v_sigmoid(self.value)
        self.output_vector = self.result[-1]
        return self.output_vector

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def back_propagation(self, data):
        # calculate the last layer's error
        self.error[-1] = (self.result[-1] - data) * self.activation_function_grad(self.value[-1])
        # calculate the other layers' error
        for layer in range(self.depth - 2, 0, -1):
            self.error[layer] = np.inner(self.weight[layer + 1], self.error[layer + 1]) \
                                * self.activation_function_grad(self.value[layer])
        # calculate the gradient of weights
        # TODO: implementation
        for layer in range(1, self.depth):
            self.weight_gradient.append(self.array2matrix(self.error[layer], self.result[layer - 1]))
        # calculate the gradient of bias:
        self.bias_gradient = self.error

    def array2matrix(self, arr1, arr2):
        x = len(arr1)
        y = len(arr2)
        mat = np.zeros((x, y))
        for j in range(x):
            for k in range(y):
                mat[j][k] = arr1[j] * arr2[k]
        return mat
