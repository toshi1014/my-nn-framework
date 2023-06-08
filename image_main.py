from PIL import Image
import cv2
import numpy as np
from my_nn_framework import models, layers, ActivationFunc


# filters
horizontal_edge_detector = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
]

vertical_edge_detector = [
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
]
# end filters


img_path = "src.png"
img = Image.open(img_path)

# raw_img_grayed = np.asarray(img.convert("L")) / 255
raw_img_grayed = np.asarray(img.convert("L"))
img_grayed = raw_img_grayed.reshape(
    raw_img_grayed.shape[0],
    raw_img_grayed.shape[1],
    -1,                         # add channel
)

model = models.CNNModel([
    layers.Conv(
        filters=(horizontal_edge_detector, vertical_edge_detector),
        kernel_size=(3, 3),
        strides=(1, 1),
        activation_func=ActivationFunc.relu,
    ),
    layers.Pool(
        kernel_size=(3, 3),
        strides=(1, 1),
        aggregation=layers.Pool.max,
    ),
    # layers.Flatten(),
    # layers.Dense(
    #     out_features=10,
    #     activation_func=ActivationFunc.softmax,
    # ),
])

out = model(img_grayed)
cv2.imwrite("out.png", out[:, :, 0])
