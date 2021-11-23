import numpy as np
from my_nn_framework import Dense, LossFunc, SGD


class Model(LossFunc):
    def __init__(self, *layer_list):
        self.layer_list = layer_list
        self.validate()

        ## init first layer
        self.layer_list[0].init_weights(self.layer_list[0].out_features)

        ## if not one layer
        if len(self.layer_list) != 1:
            for i in range(1, len(self.layer_list)):
                self.layer_list[i].init_weights(self.layer_list[i-1].out_features)

    def __repr__(self):
        return f"Model with {len(self.layer_list)} layers\n{self.layer_list}"

    def validate(self):
        if self.layer_list[0].in_features == None:
            raise Exception("No in_features in layer[0]")

        for idx, layer in enumerate(self.layer_list):
            try:
                _ = layer.out_features
            except:
                raise Exception(f"No out_features in layer[{idx}]")

    def summary(self):
        print(self)

    def compile(self, optimizer, loss):
        # TODO: add optimizers
        if optimizer == "sgd":
            self.optimizer = SGD(self.layer_list, str_loss_func=loss)

        # TODO: add losses
        if loss == "mse":
            self.loss_func = self.mes

    def fit(self, x, y, epochs, batch_size=1, learning_rate=0.001):
        assert len(x) == len(y)

        for epoch in range(epochs):
            self.optimizer(x, y, batch_size, learning_rate)