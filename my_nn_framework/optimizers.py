import random
import numpy as np


class SGD:
    def __init__(self, layer_list, str_loss_func):
        self.layer_list = layer_list
        self.str_loss_func = str_loss_func

    def __call__(self, x, y, batch_size, learning_rate):
        data_size = len(x)
        sampled_idx = random.sample(range(data_size), batch_size)
        minibatch = [[x[idx], y[idx]] for idx in sampled_idx]
        self.update_minibatch(minibatch, learning_rate)

    def loss_derivative(self, layer_idx, source, target):
        return self.layer_list[layer_idx].loss_derivative(self.str_loss_func, source, target)

    def activation_func_derivative(self, layer_idx, x):
        return self.layer_list[layer_idx].activation_func_derivative(x)

    def backprop(self,x, y):
        dl_dw = [np.zeros(layer.w.shape) for layer in self.layer_list]
        dl_db = [np.zeros(layer.b.shape) for layer in self.layer_list]

        activation = x          ## activation = activation_func(z)
        activation_list = [activation]
        z_list = []                         ## z = w @ pre_activation + b

        ## feedforward
        for layer in self.layer_list:
            z = np.dot(layer.w.transpose(), activation) + layer.b
            z_list.append(z)

            activation = layer.activation_func(z)
            activation_list.append(activation)

        ## backward
        delta = self.loss_derivative(layer_idx=-1, source=activation_list[-1], target=y) * \
            self.activation_func_derivative(-1, z_list[-1])[0]

        dl_dw[-1] = np.dot(delta, np.array(activation_list[-2]).transpose())
        dl_db[-1] = delta

        for i in range(2, len(self.layer_list)):
            z_now = z_list[-i]
            delta = np.dot(self.layer_list[-i+1].w.transpose(), delta) * \
                self.activation_func_derivative(-i, z_list[-i])

            dl_dw[-i] = np.dot(delta, activation_list[-i-1].transpose())
            dl_db[-i] = delta

        return dl_dw, dl_db


    def update_minibatch(self, minibatch, learning_rate):
        sum_dl_dw = [np.zeros(layer.w.shape) for layer in self.layer_list]
        sum_dl_db = [np.zeros(layer.b.shape) for layer in self.layer_list]

        for x, y in minibatch:
            dl_dw, dl_db = self.backprop(x, y)
            sum_dl_dw += dl_dw
            sum_dl_db += dl_db

        ## update
        for layer, dl_dw, dl_db in zip(self.layer_list, sum_dl_dw, sum_dl_db):
            layer.w -= dl_dw / len(minibatch) * learning_rate
            layer.b -= dl_db / len(minibatch) * learning_rate