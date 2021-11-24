import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from my_nn_framework import Model, Dense, to_categorical


mnist = load_digits()
data = mnist.data
label = mnist.target
x_train, x_test, y_train, y_test =  train_test_split(data, label, test_size=0.3)
y_train_one_hot = to_categorical(y_train)

in_features = x_train.shape[-1]

model = Model(
    # Dense("no", in_features=in_features, out_features=2),
    # Dense("no", out_features=10),
    Dense("sigmoid", in_features=in_features, out_features=30),
    Dense("sigmoid", out_features=10),
    # Dense("relu", in_features=3, out_features=2),
    # Dense("relu", out_features=1),
)

model.compile(optimizer="sgd", loss="mse")

model.fit(
    x_train,
    y_train_one_hot,
    epochs=300,
    batch_size=32,
    learning_rate=1
)

predicted_one_hot = model.predict(x_test)
predicted = [np.argmax(p) for p in predicted_one_hot]

accuracy = Model.evaluate(predicted, y_test)

print(f"\naccuracy: {accuracy}")

# model.show_weights()