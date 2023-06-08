import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from my_nn_framework import models, layers, ActivationFunc, \
    LossFunc, optimizers, to_categorical


def get_data():
    mnist = load_digits()
    data = mnist.data
    label = mnist.target
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, test_size=0.3
    )
    y_train_one_hot = to_categorical(y_train)

    return x_train, x_test, y_train_one_hot, y_test


def main():
    x_train, x_test, y_train_one_hot, y_test = get_data()
    in_features = x_train.shape[-1]

    model = models.Model(
        # layers.Dense("no", in_features=in_features, out_features=2),
        # layers.Dense("no", out_features=10),
        layers.Dense(
            in_features=in_features,
            out_features=30,
            activation_func=ActivationFunc.relu,
        ),
        layers.Dense(
            out_features=y_train_one_hot.shape[1],
            activation_func=ActivationFunc.sigmoid,
        ),
    )

    model.compile(optimizer=optimizers.SGD, loss=LossFunc.mse)

    model.fit(
        x_train,
        y_train_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    predicted_one_hot = model.predict(x_test)
    predicted = [np.argmax(p) for p in predicted_one_hot]

    accuracy = models.Model.evaluate(predicted, y_test)

    print(f"\naccuracy: {accuracy}")

    # model.show_weights()


if __name__ == "__main__":
    # params
    epochs = 300
    batch_size = 32
    learning_rate = 0.9

    main()
