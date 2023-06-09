import numpy as np


def get_derivative_func(activate_func):
    return derivative_pair_dict[activate_func]


class ActivationFunc:
    def __init__(self, activation_func):
        self.activation_func = activation_func
        self.activation_func_derivative = get_derivative_func(activation_func)

    @classmethod
    def no(cls, x):
        return x

    @classmethod
    def no_derivative(cls, x):
        return 1

    @classmethod
    def relu(cls, x):
        return np.array(list(
            map(lambda x0: [max(0, x0.item())], x)
        ))

    @classmethod
    def relu_derivative(cls, x):
        return np.array(list(
            map(lambda x0: [1] if (x0 > 0) else [0], x)
        ))

    @classmethod
    def sigmoid(cls, x):
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def sigmoid_derivative(cls, x):
        return cls.sigmoid(x) * (1 - cls.sigmoid(x))

    @classmethod
    def softmax(cls, x):
        return np.exp(x) / sum(np.exp(x))

    @classmethod
    def softmax_derivative(cls, x):
        ...     # dummy


derivative_pair_dict = {
    ActivationFunc.no: ActivationFunc.no_derivative,
    ActivationFunc.relu: ActivationFunc.relu_derivative,
    ActivationFunc.sigmoid: ActivationFunc.sigmoid_derivative,
    ActivationFunc.softmax: ActivationFunc.softmax_derivative,
}
