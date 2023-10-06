import pytest
import numpy as np
import matplotlib.pyplot as plt

from nexum.core.models import Perceptron


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
            image[size[1] - 1 - j][i] = color(lim(nn.predict(np.array([x, y]))[0]))

    plt.imshow(
        image,
        extent=(
            *image_range[0],
            *image_range[1],
        ),
    )


def test_split_line():
    print()

    nn = Perceptron(layers_config=[2, 1])
    # nn.calculate_err(0.32, 1)

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
        result = nn.predict(input_data[i])[0]
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

    image_range = (
        (np.min(input_data[:, 0]), np.max(input_data[:, 0])),
        (np.min(input_data[:, 1]), np.max(input_data[:, 1])),
    )

    plot_split_line_image(
        nn=nn,
        image_range=image_range,
    )

    w = nn.w[-1][0]
    bias = nn.layers[-1].bias[0]
    k = -(w[0] / w[1])
    b = -(bias / w[1])
    print(f"{k = }")
    print(f"{b = }")

    x = np.linspace(np.min(input_data[:, 0]), np.max(input_data[:, 0]), 2)
    y = np.array(list(map(lambda x: k * x + b, x)))

    plt.plot(x, y)

    if c1_points.ndim > 1:
        plt.scatter(c1_points[:, 0], c1_points[:, 1])
    if c2_points.ndim > 1:
        plt.scatter(c2_points[:, 0], c2_points[:, 1])

    plt.show()
