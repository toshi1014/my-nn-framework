import numpy as np

class ActivationFunc:
    def __init__(self, str_activation_func):
        if str_activation_func == "relu":
            self.activation_func = self.relu
            self.activation_func_derivative = self.relu_derivative

        elif str_activation_func == "sigmoid":
            self.activation_func = self.sigmoid
            self.activation_func_derivative = self.relu_derivative

        else:
            raise ValueError("Invalid activation function")

    def relu(self, x):
        return [max(0, i) for i in x]

    def relu_derivative(self, x):
        return [1 if (i>0) else 0 for i in x]

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - sigmoid(x))

    def loss_derivative(self, str_loss_func, source, target):
        if str_loss_func == "mse":
            return np.mean(np.array(source) - np.array(target))