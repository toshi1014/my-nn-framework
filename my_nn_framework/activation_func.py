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
        return np.array(
            [[max(0, i)] for i in x],
            dtype=object,
        )

    @classmethod
    def relu_derivative(cls, x):
        val = np.array(
            [[1] if (i > 0) else [0] for i in x],
            dtype=object,
        )
        return val

    @classmethod
    def sigmoid(cls, x):
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def sigmoid_derivative(cls, x):
        val = cls.sigmoid(x) * (1 - cls.sigmoid(x))
        return val


derivative_pair_dict = {
    ActivationFunc.no: ActivationFunc.no_derivative,
    ActivationFunc.relu: ActivationFunc.relu_derivative,
    ActivationFunc.sigmoid: ActivationFunc.sigmoid_derivative,
}
