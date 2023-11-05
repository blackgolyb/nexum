import numpy as np
import plotly.graph_objects as go


def get_clustered_space_splitter(nn, input_data):
    w = nn.layers[-1].w
    traces = []
    bias = nn.layers[-1].bias
    if nn.layers[-2].node_number == 2:
        for i in range(w.shape[0]):
            w_c = w[i]
            k = -(w_c[0] / w_c[1])
            b = -(bias[i][0] / w_c[1])

            name = f"{i}: y = x{k:.4f} + {b:.4f}"

            x = np.linspace(np.min(input_data[:, 0]), np.max(input_data[:, 0]), 2)
            y = k * x + b
            traces.append(go.Scatter(x=x, y=y, name=name))

    elif nn.layers[-2].node_number == 3:
        for i in range(w.shape[0]):
            w_c = w[i]
            k1 = -(w_c[0] / w_c[2])
            k2 = -(w_c[1] / w_c[2])
            b = -(bias[i][0] / w_c[2])

            xmin, xmax = np.min(input_data[:, 0]), np.max(input_data[:, 0])
            ymin, ymax = np.min(input_data[:, 1]), np.max(input_data[:, 1])

            x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

            z = k1 * x + k2 * y + b

            name = f"{i}: z = x{k1:.4f} + y{k2:.4f} + {b:.4f}"

            traces.append(go.Surface(x=x, y=y, z=z, name=name))

    return traces


def get_clustered_space_image(nn, input_data, n=200, margins=None):
    def lim(x):
        return x >= 0.5

    def get_color_function(color):
        return lambda x, n: np.clip(
            x * color / n,
            0,
            255,
        )

    xmin = np.min(input_data[:, 0])
    xmax = np.max(input_data[:, 0])
    ymin = np.min(input_data[:, 1])
    ymax = np.max(input_data[:, 1])

    margins = max((xmax - xmin), (ymax - ymin)) * 0.2
    image_range = (
        (xmin - margins, xmax + margins),
        (ymin - margins, ymax + margins),
    )

    xmin = image_range[0][0]
    xmax = image_range[0][1]
    ymin = image_range[1][0]
    ymax = image_range[1][1]

    size = (n, n)

    colors = []
    out_n = nn.layers[-1].node_number
    for i in range(out_n):
        random_color = np.random.rand(3)
        colors.append(get_color_function(random_color))

    image = np.zeros((size[1], size[0], 3))

    for i in range(size[0]):
        for j in range(size[1]):
            x = (xmax - xmin) / size[0] * i + xmin
            y = (ymax - ymin) / size[1] * j + ymin
            for result_i, result in enumerate(nn.predict(np.array([x, y]))):
                image[j][i] += colors[result_i](lim(result), out_n)

    image = (image * 255).astype(np.uint8)

    image = go.Image(
        x0=xmin,
        y0=ymin,
        dx=(xmax - xmin) / size[0],
        dy=(ymax - ymin) / size[1],
        z=image,
    )
    return image


def get_scatters_clustered(nn, input_data):
    results = []

    limit = np.vectorize(lambda x: int(x >= 0.5))

    for i in range(input_data.shape[0]):
        result = nn.predict(input_data[i])
        result = limit(result)
        results.append(result)

    c_points = []
    for i in range(2 ** len(results)):
        c_points.append([])

    for i, class_type in enumerate(results):
        point = input_data[i]

        c_id = sum([2**j * class_type[j] for j in range(len(class_type))])
        c_points[c_id].append(point)

    traces = []
    for i in range(len(c_points)):
        c_points[i] = np.array(c_points[i])
        if c_points[i].ndim <= 1:
            continue

        if input_data.shape[1] == 2:
            trace = go.Scatter(x=c_points[i][:, 0], y=c_points[i][:, 1], mode="markers")
        elif input_data.shape[1] == 3:
            trace = go.Scatter3d(
                x=c_points[i][:, 0],
                y=c_points[i][:, 1],
                z=c_points[i][:, 2],
                mode="markers",
            )
        traces.append(trace)

    return traces


def accuracy_score(a, b, normalize=True):
    acc = 0

    for i in range(a.shape[0]):
        acc += np.allclose(a[i], b[i])

    if normalize:
        acc /= a.shape[0]

    return acc
