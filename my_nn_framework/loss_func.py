import numpy as np

class LossFunc:
    def __init__(self):
        ...

    def mes(source, target):
        return np.sum(np.square(source - target))

    @classmethod
    def get_loss(cls, str_loss_func, source, target):
        if str_loss_func == "mse":
            return cls.mes(source, target)

    @classmethod
    def loss_derivative(cls, str_loss_func, source, target):
        if str_loss_func == "mse":
            return source - target