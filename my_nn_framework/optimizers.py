import random
import numpy as np
from my_nn_framework import LossFunc


class SGD:
    def __init__(self, layer_list, str_loss_func):
        self.layer_list = layer_list
        self.str_loss_func = str_loss_func

    def __call__(self, x, y, batch_size, learning_rate):
        data_size = len(x)
        sampled_idx = random.sample(range(data_size), batch_size)
        minibatch = [[x[idx], y[idx]] for idx in sampled_idx]
        loss = self.update_minibatch(minibatch, learning_rate)
        return loss

    def loss_derivative(self, layer_idx, source, target):
        return LossFunc.loss_derivative(self.str_loss_func, source, target)

    def activation_func_derivative(self, layer_idx, x):
        return self.layer_list[layer_idx].activation_func_derivative(x)

    def backprop(self, x, y):
        dl_dw = np.array([np.zeros(layer.w.shape) for layer in self.layer_list], dtype=object)
        dl_db = np.array([np.zeros(layer.b.shape) for layer in self.layer_list], dtype=object)

        ## activation = activation_func(z)
        activation = np.reshape(x, (len(x), 1))     ## make x in column vec
        y = np.reshape(y, (len(y), 1))              ## make y in column vec
        activation_list = [activation]
        z_list = []                         ## z = w @ pre_activation + b

        ## feedforward
        for layer in self.layer_list:
            z = np.dot(layer.w, activation) + layer.b
            z_list.append(z)

            activation = layer.activation_func(z)

            if np.isnan(activation).any():
                raise Exception("got nan")

            activation_list.append(activation)

        loss_now = LossFunc.get_loss(self.str_loss_func, activation, y)

        ## backward
        loss_derivatived = self.loss_derivative(layer_idx=-1, source=activation_list[-1], target=y)
        activation_func_derivatived = self.activation_func_derivative(-1, z_list[-1])
        delta = loss_derivatived * activation_func_derivatived

        val = np.dot(delta, np.array(activation_list[-2]).T)
        dl_dw[-1] = val
        dl_db[-1] = delta

        for i in range(2, len(self.layer_list)+1):
            delta = np.dot(self.layer_list[-i+1].w.T, delta) *  self.activation_func_derivative(-i, z_list[-i])
            dl_dw[-i] = np.dot(delta, activation_list[-i-1].T)
            dl_db[-i] = delta

        return dl_dw, dl_db, loss_now

    def update_minibatch(self, minibatch, learning_rate):
        sum_dl_dw = np.array([np.zeros(layer.w.shape) for layer in self.layer_list], dtype=object)
        sum_dl_db = np.array([np.zeros(layer.b.shape) for layer in self.layer_list], dtype=object)
        sum_loss_now = 0

        for x, y in minibatch:
            dl_dw, dl_db, loss_now = self.backprop(x, y)
            sum_dl_dw += dl_dw
            sum_dl_db += dl_db
            sum_loss_now += loss_now

        ## update
        for layer, dl_dw, dl_db in zip(self.layer_list, sum_dl_dw, sum_dl_db):
            layer.w -= dl_dw / len(minibatch) * learning_rate
            layer.b -= dl_db / len(minibatch) * learning_rate

        return sum_loss_now/len(minibatch)