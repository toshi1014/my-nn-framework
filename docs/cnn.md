# CNN

## Conv

* Number of hyperparameters : 4

    1. kernel size
    2. stride
    3. padding
    4. filters

* Number of parameters

$$
{kernel}^2 \cdot channel_n \cdot channel_{n+1} + channel_{n+1}
$$

* Output size

$$
size_{out} = \frac{size_{in} + 2 \cdot padding - kernel}{stride} + 1
$$

## Pooling

* Number of hyperparameters : 3

    1. kernel size
    2. stride
    3. padding

* Number of parameters : 0

## Fully-Connected

* Number of hyperparameters : 1

    1. Neurons in layer

* Number of parameters

$$
units_n \cdot units_{n + 1} + units_{n + 1}
$$
