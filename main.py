from my_nn_framework import Model, Dense

x_train = [
    [1,2,3],
    [4,5,6],
]

y_train = [
    [11],
    [21],
]

model = Model(
    Dense("relu", in_features=3, out_features=2),
    Dense("relu", out_features=1),
)

model.compile(optimizer="sgd", loss="mse")

model.fit(x_train, y_train, epochs=5, batch_size=1)


import pdb; pdb.set_trace()