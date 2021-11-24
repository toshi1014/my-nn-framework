import numpy as np
from my_nn_framework import Model, Dense

x_train = [
    [1,2,3],
    [11,12,13],
    [21,22,23],
]

x_test = [
    [4,5,6],
]

y_train = [
    [4],
    [5],
    [6],
]

y_test = [
    [21],
]

model = Model(
    # Dense("no", in_features=3, out_features=2),
    # Dense("no", out_features=1),
    Dense("sigmoid", in_features=3, out_features=2),
    Dense("sigmoid", out_features=1),
    # Dense("relu", in_features=3, out_features=2),
    # Dense("relu", out_features=1),
)

model.compile(optimizer="sgd", loss="mse")

model.fit(x_train, y_train, epochs=1000, batch_size=1, learning_rate=0.01)

predicted = model.predict(x_train)

accuracy = Model.evaluate(predicted, y_train)

print(f"\naccuracy: {accuracy}")

# model.show_weights()