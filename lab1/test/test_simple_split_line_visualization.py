import pytest
import numpy as np
import matplotlib.pyplot as plt

from services.neural_network import NeuralNetwork


def nne(values, w1, w2, w3):
    result = []
    activation_function = lambda x: x >= 0.5
    activation_function = lambda x: 1 / (1 + np.exp(-x))

    for value in values:
        node = value[0] * w1 + value[1] * w2 + w3
        result.append(activation_function(node))

    return result


def test_split_line():
    nn = NeuralNetwork(layers_config=[2, 1])

    input_data = np.array(
        [
            [20, -10],
            [0, 0],
            [1, -10],
            [4, 10],
            [-20, -10],
        ]
    )

    results = []

    for i in range(input_data.shape[0]):
        result = nn.feat_forward(input_data[i])[0]
        result = int(result >= 0.5)
        print(f"data: {input_data[i]}  {result=}")
        results.append(result)

    w = nn.w[-1][0]

    print(nne(input_data, w[0], w[1], w[2]))

    print(w)

    k = -(w[0] / w[1])
    b = -(w[2] / w[1])
    print(f"{k = }")
    print(f"{b = }")

    x = np.linspace(np.min(input_data[:, 0]), np.max(input_data[:, 0]), 2)
    y = np.array(list(map(lambda x: k * x + b, x)))

    plt.plot(x, y)

    c1_points = []
    c2_points = []

    for i, class_type in enumerate(results):
        point = input_data[i]

        if class_type == 0:
            c1_points.append(point)
        elif class_type == 1:
            c2_points.append(point)

    c1_points = np.array(c1_points)
    c2_points = np.array(c2_points)

    if c1_points.ndim > 1:
        plt.scatter(c1_points[:, 0], c1_points[:, 1])
    if c2_points.ndim > 1:
        plt.scatter(c2_points[:, 0], c2_points[:, 1])

    plt.show()
