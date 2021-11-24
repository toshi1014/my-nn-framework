import numpy as np
import matplotlib.pyplot as plt
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

    def show_weights(self):
        for layer in self.layer_list:
            print("=================================")
            print("weight:\n", layer.w)
            print("\n\nbias:\n", layer.b)
            print("\n\nactivation_func:\n", layer.activation_func)
        print("=================================")

    def compile(self, optimizer, loss):
        # TODO: add optimizers
        if optimizer == "sgd":
            self.optimizer = SGD(self.layer_list, str_loss_func=loss)

        # TODO: add losses
        if loss == "mse":
            self.loss_func = self.mes

    def fit(self, x, y, epochs, batch_size=1, learning_rate=0.001):
        assert len(x) == len(y)

        loss_list = []

        for epoch in range(epochs):
            loss = self.optimizer(x, y, batch_size, learning_rate)
            print(f"epoch {epoch}")
            print(f"loss: {loss}")
            loss_list.append(loss)

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(range(len(loss_list)), loss_list)
        plt.savefig("out.png")

    def feedforward(self, x):
        activation = np.reshape(x, (len(x), 1))     ## make x in column vec
        for layer in self.layer_list:
            z = np.dot(layer.w, activation) + layer.b
            activation = layer.activation_func(z)
        return activation.reshape(-1)       ## into 1D vector

    def predict(self, x_test):
        predicted_list = []
        for x in x_test:
            predicted = self.feedforward(x)
            predicted_list.append(predicted)
        return predicted_list

    @classmethod
    def evaluate(cls, predicted, y_test):
        assert len(predicted) == len(y_test)
        correct_or_wrong_list = []
        for p, y in zip(predicted, y_test):
            if p == y:
                correct_or_wrong = 1
            else:
                correct_or_wrong = 0
            correct_or_wrong_list.append(correct_or_wrong)

        return sum(correct_or_wrong_list)/len(correct_or_wrong_list)