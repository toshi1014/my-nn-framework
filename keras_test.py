import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras as K

mnist = load_digits()
data = mnist.data
label = mnist.target
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)
y_train = K.utils.to_categorical(y_train)
y_test = K.utils.to_categorical(y_test)

input_shape = x_train.shape[-1]

model = K.Sequential([
    K.layers.Dense(units=30, activation="sigmoid"),
    K.layers.Dense(units=10, activation="sigmoid"),
])

opt = K.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=opt, loss="mse")

model.fit(x_train, y_train, epochs=100, batch_size=32)

predicted_one_hot = model.predict(x_train)
predicted = [np.argmax(p) for p in predicted_one_hot]
y_train = [np.argmax(p) for p in y_train]
accuracy = accuracy_score(predicted, y_train)
print(f"accuracy: {accuracy}")
