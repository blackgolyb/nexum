import pytest
import numpy as np
import matplotlib.pyplot as plt

from neural_network.core.neural_network import Perceptron
from neural_network.core.layers import OutputLayer


def plot_split_line_image(nn, image_range, n=10, margins=2):
    image_range = (
        (image_range[0][0] - margins, image_range[0][1] + margins),
        (image_range[1][0] - margins, image_range[1][1] + margins),
    )

    xmin = image_range[0][0]
    xmax = image_range[0][1]
    ymin = image_range[1][0]
    ymax = image_range[1][1]

    size = (
        int(xmax - xmin) * n,
        int(ymax - ymin) * n,
    )

    color = lambda x: np.clip(
        1 - x * (255 - np.array([8, 8, 143])) / 255,
        0,
        255,
    )
    color = lambda x: np.clip(
        x * np.array([8, 8, 143]) / 255,
        0,
        255,
    )
    lim = lambda x: x >= 0.5

    image = np.empty((size[1], size[0], 3))

    for i in range(size[0]):
        for j in range(size[1]):
            x = (xmax - xmin) / size[0] * i + xmin
            y = (ymax - ymin) / size[1] * j + ymin
            image[size[1] - 1 - j][i] = color(lim(nn.feat_forward(np.array([x, y]))[0]))

    plt.imshow(
        image,
        extent=(
            *image_range[0],
            *image_range[1],
        ),
    )


def test_backprop():
    print()

    # o_l = OutputLayer(1, activation_function="ReLu")
    # nn = Perceptron(layers_config=[2, o_l])
    nn = Perceptron(layers_config=[2, 1])

    input_data = np.array(
        [
            [1, 1],
            [1, -1],
            [0.5, 0.5],
            [-1, 1],
            [-1, -1],
            [-0.5, -0.5],
        ]
    )

    # input_data = np.ones((1000, 2))

    input_data[:, 0] -= 10
    input_data[:, 1] -= 5

    # input_data
    x_min = np.min(input_data[:, 0])
    x_max = np.max(input_data[:, 0])
    y_min = np.min(input_data[:, 1])
    y_max = np.max(input_data[:, 1])

    x_c = (x_max + x_min) / 2
    y_c = (y_max + y_min) / 2
    # input_data[:, 0] -= x_min
    # input_data[:, 1] -= y_min
    # input_data[:, 0] /= x_max - x_min
    # input_data[:, 1] /= y_max - y_min

    targets = np.array(
        [
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
        ]
    )

    # targets = np.ones((1000, 1))
    prev_w = nn.w[0].copy()
    prev_bias = nn.layers[-1].bias[0]
    nn.train(input_data, targets, epochs=10000, learning_rate=0.01)
    print(prev_w)
    print(prev_bias)
    bias = nn.layers[-1].bias[0]
    print(nn.w)
    print(bias)
    # return
    # w = nn.w[-1][0]
    # print(nn.w)

    # k = -(w[0] / w[1])
    # b = -(bias / w[1])
    # # b = y_c - k * x_c
    # print(f"{k = }")
    # print(f"{b = }")

    # x = np.linspace(x_min, x_max, 2)
    # y = np.array(list(map(lambda x: k * x + b, x)))

    # plt.plot(x, y)

    # results = []

    # for i in range(input_data.shape[0]):
    #     result = nn.feat_forward(input_data[i])[0]
    #     result = int(result >= 0.5)
    #     print(f"data: {input_data[i]}  {result=}")
    #     results.append(result)

    image_range = (
        (np.min(input_data[:, 0]), np.max(input_data[:, 0])),
        (np.min(input_data[:, 1]), np.max(input_data[:, 1])),
    )

    plot_split_line_image(
        nn=nn,
        image_range=image_range,
    )

    results = []

    for i in range(input_data.shape[0]):
        result = nn.feat_forward(input_data[i])[0]
        result = int(result >= 0.5)
        print(f"data: {input_data[i]}  {result=}")
        results.append(result)

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
    # nn.calculate_err(
    #     np.array(
    #         [
    #             0.32,
    #         ]
    #     ),
    #     np.array(
    #         [
    #             1,
    #         ]
    #     ),
    # )
