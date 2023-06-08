import numpy as np
from my_nn_framework import ActivationFunc


class Layer(ActivationFunc):
    def __init__(self, activation_func, in_features, out_features):
        super(Layer, self).__init__(activation_func=activation_func)
        self.in_features = in_features
        self.out_features = out_features

    def init_weights(self, in_features):
        if self.in_features is None:
            self.in_features = in_features

        self.w = np.random.normal(size=(self.out_features, self.in_features))
        self.b = np.random.normal(size=(self.out_features, 1))

    def __call__(self, x):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, activation_func, out_features, in_features=None):
        super(Dense, self).__init__(
            activation_func,
            in_features,
            out_features,
        )

    def __call__(self, x):
        super(Dense, self).init_weights(x.shape[0])

        z = np.dot(self.w, x) + self.b
        return self.activation_func(z)

    def __repr__(self):
        return f"[Dense: input_features={self.in_features}, out_features={self.out_features}]"

    def init_weights(self, in_features):
        super().init_weights(in_features)


class SubMatrixOperator:
    def __init__(self, kernel_size, strides):
        self.kernel_size = kernel_size
        self.strides = strides

    def __call__(self, x, func):
        output = []
        proper_sub_matrix_size = self.kernel_size[0] * \
            self.kernel_size[1] * x.shape[2]

        for i in range(0, x.shape[0], self.strides[0]):
            output_row = []
            for j in range(0, x.shape[1], self.strides[1]):
                sub_matrix = x[
                    i:self.kernel_size[0] + i,
                    j:self.kernel_size[1] + j
                ]

                if sub_matrix.size == proper_sub_matrix_size:
                    output_row.append(func(sub_matrix))

            if len(output_row) != 0:
                output.append(output_row)

        return np.array(output)


class Conv(SubMatrixOperator):
    def __init__(
            self, filters, kernel_size,
            strides, activation_func,
    ):
        super().__init__(kernel_size, strides)
        self.filters = filters
        self.activation_func = activation_func
        self.b = np.random.rand(len(self.filters))

    def __call__(self, x):
        return super().__call__(
            x,
            lambda sub_matrix: [
                np.sum([
                    self.activation_func(
                        [np.sum(filter * sub_matrix[:, :, idx_channel]) + bias]
                    )[0]        # TEMP: 
                    for idx_channel in range(sub_matrix.shape[2])
                ])
                for filter, bias in zip(self.filters, self.b)
            ]
        )


class Pool(SubMatrixOperator):
    def __init__(self, kernel_size, strides, aggregation):
        super().__init__(kernel_size, strides)
        self.aggregation = aggregation

    def __call__(self, x):
        return super().__call__(x, self.aggregation)

    @staticmethod
    def wrapper(sub_matrix, func):
        return [
            func(sub_matrix[:, :, idx_channel])
            for idx_channel in range(sub_matrix.shape[2])
        ]

    @staticmethod
    def max(sub_matrix):
        return Pool.wrapper(sub_matrix, np.max)

    @staticmethod
    def min(sub_matrix):
        return Pool.wrapper(sub_matrix, np.min)

    @staticmethod
    def avg(sub_matrix):
        return Pool.wrapper(sub_matrix, np.mean)


class Flatten:
    def __init__(self):
        ...

    def __call__(self, x):
        return x.reshape(-1)
