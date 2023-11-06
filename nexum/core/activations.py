from abc import ABC, abstractstaticmethod
from enum import Enum
from typing import Callable, NoReturn

import numpy as np

from nexum.core.initializations import (
    InitializationFunctions,
    get_initialization_function_by_enum,
)
from nexum.services.enums import ContainsEnumMeta


class ActivationFunctions(str, Enum, metaclass=ContainsEnumMeta):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    HTAN = "htan"
    RELU = "relu"
    SOFTMAX = "softmax"


class ABCActivationFunction(ABC):
    best_init_functions: list[Callable] = []

    @abstractstaticmethod
    def activation_function(x):
        raise NotImplementedError()

    @abstractstaticmethod
    def derivation_of_activation_function(x):
        raise NotImplementedError()

    def get_functions(self):
        return (self.activation_function, self.derivation_of_activation_function)

    def get_best_init_functions(self):
        return self.best_init_functions


class BaseActivationFunction(ABCActivationFunction):
    def calculate(self, input_data):
        self.input = input_data
        return self.activation_function(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(
            output_gradient,
            self.derivation_of_activation_function(self.input),
        )


class Linear(BaseActivationFunction):
    best_init_functions = [
        get_initialization_function_by_enum(InitializationFunctions.XAVIER),
        get_initialization_function_by_enum(InitializationFunctions.RANDOM_2),
    ]

    @staticmethod
    def activation_function(x):
        return x

    @staticmethod
    def derivation_of_activation_function(x):
        return 1


class Sigmoid(BaseActivationFunction):
    best_init_functions = [
        get_initialization_function_by_enum(InitializationFunctions.XAVIER),
        get_initialization_function_by_enum(InitializationFunctions.RANDOM_2),
    ]

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivation_of_activation_function(x):
        s = Sigmoid().activation_function(x)
        return s * (1 - s)


class ReLu(BaseActivationFunction):
    best_init_functions = [
        get_initialization_function_by_enum(InitializationFunctions.XAVIER),
    ]

    @staticmethod
    def activation_function(x):
        return np.maximum(x, np.zeros(x.shape))

    @staticmethod
    def derivation_of_activation_function(x):
        return np.vectorize(lambda x: 0 if x < 0 else 1)(x)


class Softmax(BaseActivationFunction):
    best_init_functions = [
        get_initialization_function_by_enum(InitializationFunctions.XAVIER),
        get_initialization_function_by_enum(InitializationFunctions.RANDOM_2),
    ]

    @staticmethod
    def activation_function(x):
        ...

    @staticmethod
    def derivation_of_activation_function(x):
        ...

    def calculate(self, input_nodes):
        tmp = np.exp(input_nodes)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


class CustomActivationFuncHasNoInitializationFuncError(ValueError):
    message = (
        "When you use own activation function "
        "you must setup the initialization function by your self"
    )

    def __init__(self, message=None):
        self.message = message or self.message
        super().__init__(self.message)


class Custom(BaseActivationFunction):
    def __init__(self, activation_function, derivation_activation):
        self.activation_function = activation_function
        self.derivation_of_activation_function = derivation_activation

    def get_best_init_functions(self) -> NoReturn:
        raise CustomActivationFuncHasNoInitializationFuncError()

    @staticmethod
    def activation_function(x):
        ...

    @staticmethod
    def derivation_of_activation_function(x):
        ...


activation_function_by_enum = {
    ActivationFunctions.LINEAR: Linear,
    ActivationFunctions.SIGMOID: Sigmoid,
    ActivationFunctions.RELU: ReLu,
    ActivationFunctions.SOFTMAX: Softmax,
}


def get_activation_function_by_enum(enum_val: InitializationFunctions):
    return activation_function_by_enum[enum_val]()
