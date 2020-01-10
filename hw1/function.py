# Created by haiphung106

import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def sigmoid_delta(a):
    return sigmoid(a) * (1 - sigmoid(a))


def softmax(a):
    tmp = np.exp(a)
    Z = tmp / tmp.sum(axis=1, keepdims=True)
    return Z


def linear(a):
    return a


def linear_delta(a):
    # return np.ones((a.shape))
    return 1


def activation_function(x, activation):
    if activation == 'sigmoid':
        return sigmoid(x)
    elif activation == 'softmax':
        return softmax(x)
    elif activation == 'linear':
        return linear(x)


def delta_function(x, activation):
    if activation == 'sigmoid':
        return sigmoid_delta(x)
    elif activation == 'softmax':
        return None
    elif activation == 'linear':
        return linear_delta(x)


class NeuralNetwork:
    def __init__(self, layers, activation, LR):
        self.layers = layers
        self.activation = activation
        self.LR = LR
        self.number_layer = len(layers) - 1
        self.weight = []
        self.bias = []
        self.t = []
        self.z = []
        self.weight_delta = []
        self.bias_delta = []
        self.dt = []

    def init_weight(self):

        np.random.seed(3)

        for i in range(self.number_layer):
            weight = np.random.randn(self.layers[i], self.layers[i + 1])
            bias = np.zeros([self.layers[i + 1], 1])
            self.weight.append(weight)
            self.bias.append(bias)

    def feedforward(self, x):
        self.t = []
        self.z = []
        self.t.append(x)
        for i in range(self.number_layer):
            input_layer = self.t[i]
            z = np.dot(input_layer, self.weight[i]) + self.bias[i].T
            self.z.append(z)
            self.t.append(activation_function(z, self.activation[i]))

    def backpropagation(self, x, t):
        self.bias_delta = []
        self.weight_delta = []
        self.dt = []

        t = np.array(t).reshape(-1, 1)
        t_l = self.t[-1]
        t_l_delta = 2 * (t_l - t)
        # print(t_l_delta.shape)

        self.dt.append(t_l_delta)

        for i in reversed(range(self.number_layer)):
            t_l = self.t[i + 1]
            z_l = self.z[i]
            t_l_delta = self.dt[-1]
            t_l_z_delta = delta_function(z_l, self.activation[i])
            w_l = self.weight[i]

            self.weight_delta.append(np.dot(self.t[i].T, t_l_delta * t_l_z_delta))
            self.bias_delta.append(np.sum(t_l_delta * t_l_z_delta, 0).reshape(-1, 1))
            self.dt.append(np.dot((t_l_delta * t_l_z_delta), w_l.T))

        self.weight_delta = self.weight_delta[::-1]
        self.bias_delta = self.bias_delta[::-1]

    def weight_update(self):
        for i in range(self.number_layer):
            self.weight[i] = self.weight[i] - self.LR * self.weight_delta[i]
            self.bias[i] = self.bias[i] - self.LR * self.bias_delta[i]

    def predict(self, x):
        input_layer = np.array(x, dtype=float)
        for i in range(self.number_layer):
            z = np.dot(input_layer, self.weight[i]) + self.bias[i].T
            input_layer = activation_function(z, self.activation[i])
        return input_layer.reshape(-1, 1)

    def backpropagation_classification(self, x, t):
        self.bias_delta = []
        self.weight_delta = []
        self.dt = []

        t_l = self.t[-1]
        w_l = self.weight[-1]
        z_l_delta = t_l - t
        # print(z_l_delta)
        w_l_delta = np.dot(self.t[self.number_layer - 1].T, z_l_delta)
        bias_l_delta = np.sum(z_l_delta, 0).reshape(-1, 1)

        self.dt.append(np.dot(z_l_delta, w_l.T))
        self.weight_delta.append(w_l_delta)
        self.bias_delta.append(bias_l_delta)

        for i in reversed(range(0, self.number_layer - 1)):
            t_l = self.t[i + 1]
            z_l = self.z[i]
            t_l_delta = self.dt[-1]
            t_l_z_delta = delta_function(z_l, self.activation[i])
            w_l = self.weight[i]

            self.weight_delta.append(np.dot(self.t[i].T, t_l_delta * t_l_z_delta))
            self.bias_delta.append(np.sum(t_l_delta * t_l_z_delta, 0).reshape(-1, 1))
            self.dt.append(np.dot((t_l_delta * t_l_z_delta), w_l.T))

        self.weight_delta = self.weight_delta[::-1]
        self.bias_delta = self.bias_delta[::-1]

    def predict_y(self, x):
        input_layer = np.array(x, dtype=float)
        for i in range(self.number_layer):
            z = np.dot(input_layer, self.weight[i]) + self.bias[i].T
            input_layer = activation_function(z, self.activation[i])
        return input_layer.reshape(len(x), 2)

    def cross_entropy(self, x, t):
        pred_y = self.predict_y(x)
        cross_error = 0
        for i in range(t.shape[0]):
            index = np.argmax(t[i], axis=0)
            cross_error = cross_error - np.log(pred_y[i][index])
        return cross_error

    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))


def Onehotvector(x):
    lookup = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
        dtype=float)
    b = []
    for i in range(len(x)):
        if int(x[i]) == 0:
            b.append(lookup[0])
        elif int(x[i]) == 1:
            b.append(lookup[1])
        elif int(x[i]) == 2:
            b.append(lookup[2])
        elif int(x[i]) == 3:
            b.append(lookup[3])
        elif int(x[i]) == 4:
            b.append(lookup[4])
        else:
            b.append(lookup[5])
    b = np.reshape(b, (len(x), 5))
    return b


def count_diff(A, B):
    n = len(A)
    C = A * B
    return n - np.sum(C)


def convertBinary(x):
    loockup = np.array(([0, 1], [1, 0]), dtype=int)
    return loockup[int(x)]
