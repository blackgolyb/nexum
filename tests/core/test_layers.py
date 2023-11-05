import numpy as np

from nexum.core.activations import (
    ActivationFunctions,
    Custom,
    CustomActivationFuncHasNoInitializationFuncError,
    Sigmoid,
)
from nexum.core.initializations import InitializationFunctions, xavier_init
from nexum.core.layers import FullConnectedLayer, OutputLayer


def test_default_creation():
    FullConnectedLayer(10)


def test_activation_function_enum_param():
    layer = FullConnectedLayer(10, activation_function=ActivationFunctions.SIGMOID)
    assert isinstance(layer.activation_function_obj, Sigmoid)


def test_activation_function_obj_param():
    layer = FullConnectedLayer(10, activation_function=Sigmoid())
    assert isinstance(layer.activation_function_obj, Sigmoid)


def test_activation_function_string_param():
    layer = FullConnectedLayer(10, activation_function="sigmoid")
    assert isinstance(layer.activation_function_obj, Sigmoid)


def test_activation_function_lambda_param1():
    is_error = False
    try:
        FullConnectedLayer(10, activation_function=(lambda x: x, lambda x: 1))
    except CustomActivationFuncHasNoInitializationFuncError as e:
        print(e)
        is_error = True

    assert is_error


def test_activation_function_lambda_param2():
    layer = FullConnectedLayer(
        10,
        activation_function=(lambda x: x, lambda x: 1),
        initialization_function=InitializationFunctions.RANDOM_1,
    )

    assert isinstance(layer.activation_function_obj, Custom)


def test_initialization_function_enum_param():
    layer = FullConnectedLayer(
        10,
        initialization_function=InitializationFunctions.XAVIER,
    )

    assert layer.initialization_function == xavier_init


def test_initialization_function_str_param():
    layer = FullConnectedLayer(
        10,
        initialization_function="xavier",
    )

    assert layer.initialization_function == xavier_init


def test_initialization_function_lambda_param():
    def init_func(n, m):
        return np.ones((n, m))

    layer = FullConnectedLayer(
        10,
        initialization_function=init_func,
    )

    assert layer.initialization_function == init_func


def test_layers_connecting():
    layer1 = FullConnectedLayer(10)
    layer2 = FullConnectedLayer(
        5, initialization_function=lambda n, m: np.full((n, m), 1)
    )

    layer2.connect_to_layer(layer1)

    assert layer2.w.shape == (5, 10)
    assert np.array_equal(layer2.w, np.full((5, 10), 1))


def test_bias_default_connecting():
    layer = FullConnectedLayer(10)

    assert layer.bias.shape == (10, 1)


def test_output_layer():
    o1 = OutputLayer(10, connection_type="full_connected")
    o2 = OutputLayer(10, connection_type="pair_connected")
    o3 = OutputLayer(10, connection_type="triple_connected")

    assert o1.calculate != o2.calculate
    assert o1.calculate != o3.calculate
    assert o2.calculate != o3.calculate
