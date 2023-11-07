import os

import numpy as np

from nexum.core.models import Perceptron


def test_load():
    config = [1, 2, 3, 1]
    nn = Perceptron(config)
    print(f"weights of last layer before load: {nn.layers[-1].w}")

    nn.save("test.hdf5")
    assert os.path.exists("test.hdf5")

    nn_load = Perceptron(config)
    nn_load.load("test.hdf5")
    print(f"weights of last layer after load: {nn.layers[-1].w}")

    for i in range(1, len(nn.layers)):
        assert np.array_equal(nn.layers[i].w, nn_load.layers[i].w)

    os.remove("test.hdf5")
