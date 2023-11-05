import numpy as np
import plotly

from nexum.core.layers import OutputLayer
from nexum.core.models import Perceptron
from nexum.services.utils import (
    get_clustered_space_image,
    get_clustered_space_splitter,
    get_scatters_clustered,
)


def test_backprop():
    print()

    o_l = OutputLayer(1)
    nn = Perceptron(layers_config=[2, o_l])
    nn.finalize = np.vectorize(lambda x: 0 if x <= 0.5 else 1)

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

    input_data[:, 0] -= 10
    input_data[:, 1] -= 5

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

    nn.train(input_data, targets, epochs=10000, learning_rate=0.1)

    image = get_clustered_space_image(nn, input_data)
    traces = get_scatters_clustered(nn, input_data)
    split_traces = get_clustered_space_splitter(nn, input_data)

    fig = plotly.graph_objs.Figure([image, *traces, *split_traces])
    fig.show()
