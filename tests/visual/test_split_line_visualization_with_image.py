import numpy as np
import plotly

from nexum.core.models import Perceptron

from nexum.services.utils import (
    get_clustered_space_image,
    get_clustered_space_splitter,
    get_scatters_clustered,
)


def test_split_line():
    print()

    nn = Perceptron(layers_config=[2, 2])

    input_data = np.array(
        [
            (0.5, 1.3),
            (0.8, 1.6),
            (0.9, 1.8),
            (1.2, 0.6),
            (1.5, 0.8),
            (0.2, 0.1),
            (0.1, 0.5),
            (-0.2, 0.6),
            (-0.6, -0.8),
            (-1.0, -1.2),
        ]
    )

    image = get_clustered_space_image(nn, input_data)
    traces = get_scatters_clustered(nn, input_data)
    split_traces = get_clustered_space_splitter(nn, input_data)

    fig = plotly.graph_objs.Figure([image, *traces, *split_traces])
    fig.show()
