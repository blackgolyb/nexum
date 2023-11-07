import os

from nexum.core.models import Perceptron


def test_save():
    nn = Perceptron([1, 2, 3, 1])
    nn.save("test.hdf5")
    assert os.path.exists("test.hdf5")
    os.remove("test.hdf5")
