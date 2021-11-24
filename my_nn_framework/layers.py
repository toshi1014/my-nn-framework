import numpy as np
from my_nn_framework import ActivationFunc


class Layer(ActivationFunc):
    def __init__(self, str_activation_func, in_features, out_features):
        super(Layer, self).__init__(str_activation_func=str_activation_func)
        self.in_features = in_features
        self.out_features = out_features

    def init_weights(self, in_features):
        if self.in_features == None:
            self.in_features = in_features

        self.w = np.random.normal(size=(self.out_features, self.in_features))
        self.b = np.random.normal(size=(self.out_features, 1))
        # self.w = self.w.astype(object)
        # self.b = self.b.astype(object)

    def __call__(self):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, str_activation_func, out_features, in_features=None):
        super(Dense, self).__init__(str_activation_func, in_features, out_features)

    def __call__(self):
        ...

    def __repr__(self):
        return f"[Dense: input_features={self.in_features}, out_features={self.out_features}]"

    def init_weights(self, in_features):
        super().init_weights(in_features)