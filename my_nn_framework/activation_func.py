import numpy as np

class ActivationFunc:
    def __init__(self, str_activation_func):
        if str_activation_func == "relu":
            self.activation_func = self.relu
            self.activation_func_derivative = self.relu_derivative
        elif str_activation_func == "sigmoid":
            self.activation_func = self.sigmoid
            self.activation_func_derivative = self.sigmoid_derivative
        elif str_activation_func == "no":
            self.activation_func = self.no
            self.activation_func_derivative = self.no_derivative
        else:
            raise ValueError("Invalid activation function")

    def no(self, x):
        return x

    def no_derivative(self, x):
        return 1

    def relu(self, x):
        # return np.array([[max(0, i)] for i in x], dtype=object)
        return np.array([[max(0, i)] for i in x], dtype=object)

    def relu_derivative(self, x):
        val = np.array([[1] if (i>0) else [0] for i in x], dtype=object)
        # import pdb; pdb.set_trace()
        return val

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        val =  self.sigmoid(x) * (1 - self.sigmoid(x))
        # import pdb; pdb.set_trace()
        return val

    def loss_derivative(self, str_loss_func, source, target):
        if str_loss_func == "mse":
            return source - target