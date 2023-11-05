from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Iterable, NoReturn, Self

import numpy as np

from nexum.core.activation_functions import (
    ABCActivationFunction,
    ActivationFunctions,
    Custom,
    get_activation_function_by_enum,
)
from nexum.core.initialization_functions import (
    InitializationFunctions,
    get_initialization_function_by_enum,
)
from nexum.services.enums import ContainsEnumMeta


class ABCLayer(ABC):
    def __init__(self):
        self._save_data = False

    @property
    def save_data(self):
        return self._save_data

    @save_data.setter
    def save_data(self, value: bool):
        self._save_data = value

        if not self._save_data:
            self.nodes = None


class BaseLayer(ABCLayer):
    def __init__(
        self,
        node_number: int,
        bias: int | np.ndarray | None = 1,
        activation_function=ActivationFunctions.SIGMOID,
        initialization_function=None,
    ):
        super().__init__()
        self.node_number = node_number

        self.__init_activation_function(activation_function)
        self.__init_initialization_function(initialization_function)
        self.init_bias(bias)

    def init_bias(self, bias: int | np.ndarray | None = None):
        if bias is None:
            self.bias = None
        elif isinstance(bias, int):
            self.bias = self.initialization_function(self.node_number, 1)
        elif isinstance(bias, np.array):
            self.bias = bias

    def __init_activation_function(self, activation_function):
        function_obj: ABCActivationFunction | None = None

        if activation_function is None:
            function_obj = get_activation_function_by_enum(ActivationFunctions.SIGMOID)

        elif isinstance(activation_function, ABCActivationFunction):
            function_obj = activation_function

        elif isinstance(activation_function, ActivationFunctions):
            function_obj = get_activation_function_by_enum(activation_function)

        elif isinstance(activation_function, str):
            assert activation_function in ActivationFunctions

            function_obj = get_activation_function_by_enum(
                ActivationFunctions(activation_function)
            )

        elif isinstance(activation_function, Iterable):
            assert len(activation_function) == 2

            function_obj = Custom(
                activation_function=activation_function[0],
                derivation_activation=activation_function[1],
            )

        else:
            raise ValueError()

        self.activation_function_obj = function_obj
        self.activation_function, self.train_function = function_obj.get_functions()

    def __init_initialization_function(self, initialization_function):
        function: Callable | None = None

        if isinstance(initialization_function, Callable):
            function = initialization_function

        elif isinstance(initialization_function, InitializationFunctions):
            function = get_initialization_function_by_enum(initialization_function)

        elif isinstance(initialization_function, str):
            function = get_initialization_function_by_enum(
                InitializationFunctions(initialization_function)
            )

        elif initialization_function is None:
            function = self.activation_function_obj.get_best_init_functions()[0]

        self.initialization_function = function

    def init_w(self):
        self.w = self.initialization_function(
            self.node_number, self.parent_layer.node_number
        )

    def connect_to_layer(self, layer: ABCLayer):
        self.parent_layer = layer
        self.init_w()

    @abstractmethod
    def calculate(self, input_nodes):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError()


class FullConnectedLayer(BaseLayer):
    def calculate(self, input_nodes):
        self.input = input_nodes
        signal = self.w @ self.input + self.bias
        result = self.activation_function_obj.calculate(signal)

        return result

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation_function_obj.backward(
            output_gradient, learning_rate
        )

        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.w.T @ output_gradient

        self.w -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient[0]

        return input_gradient


class PairConnectedLayer(BaseLayer):
    def calculate(self):
        print("2")


class TripleConnectedLayer(BaseLayer):
    def calculate(self):
        print("3")


class InputLayer(ABCLayer):
    def __init__(self, node_number: int):
        self.node_number = node_number

    def calculate(self, input_nodes):
        self.nodes = input_nodes
        return self.nodes

    def connect_to_layer(self, layer: ABCLayer) -> NoReturn:
        raise RuntimeError("Can not connect input layer to another")

    def backward(self, output_gradient, learning_rate):
        ...


class ConnectionTypes(str, Enum, metaclass=ContainsEnumMeta):
    FULL_CONNECTED = "full_connected"
    PAIR_CONNECTED = "pair_connected"
    TRIPLE_CONNECTED = "triple_connected"


class ABCOutputLayer(BaseLayer):
    ...


class OutputLayer(object):
    connection_type_to_cls = {
        ConnectionTypes.FULL_CONNECTED: FullConnectedLayer,
        ConnectionTypes.PAIR_CONNECTED: PairConnectedLayer,
        ConnectionTypes.TRIPLE_CONNECTED: TripleConnectedLayer,
    }

    def __new__(
        cls,
        *args,
        connection_type=ConnectionTypes.FULL_CONNECTED,
        **kwargs,
    ) -> Self:
        output_cls = cls.create_output_cls(connection_type)
        instance = output_cls(*args, **kwargs)

        return instance

    @classmethod
    def create_output_cls(cls, connection_type):
        enum_val = None

        if isinstance(connection_type, ConnectionTypes):
            enum_val = connection_type
        elif isinstance(connection_type, str):
            enum_val = ConnectionTypes(connection_type)
        else:
            raise ValueError(
                "connection_type must be a string or ConnectionTypes instance"
            )

        connection_type_layer_cls = cls.connection_type_to_cls[enum_val]
        return type("Output", (ABCOutputLayer, connection_type_layer_cls), {})


class LayerCluster(BaseLayer):
    ...
