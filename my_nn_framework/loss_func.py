import numpy as np


def get_loss_derivative_func(loss_func):
    return loss_derivative_pair_dict[loss_func]


class LossFunc:
    def __init__(self):
        ...

    def loss_derivative_func(loss_func, source, target):
        func = get_loss_derivative_func(loss_func)
        return func(source, target)

    @classmethod
    def mse(cls, source, target):
        return np.sum(np.square(source - target))

    @classmethod
    def mse_derivative(cls, source, target):
        return source - target


loss_derivative_pair_dict = {
    LossFunc.mse: LossFunc.mse_derivative,
}
