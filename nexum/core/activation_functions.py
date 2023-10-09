from enum import Enum
from abc import ABC, abstractstaticmethod
from typing import Callable, NoReturn

import numpy as np

from nexum.services.enums import ContainsEnumMeta
from nexum.core.initialization_functions import (
    InitializationFunctions,
    get_initialization_function_by_enum,
)


class ActivationFunctions(str, Enum, metaclass=ContainsEnumMeta):
    SIGMOID = "sigmoid"
    HTAN = "htan"
    RELU = "ReLu"


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


class Sigmoid(ABCActivationFunction):
    best_init_functions = [
        get_initialization_function_by_enum(InitializationFunctions.XAVIER),
        get_initialization_function_by_enum(InitializationFunctions.RANDOM_2),
    ]

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivation_of_activation_function(x):
        return x * (1 - x)


class ReLu(ABCActivationFunction):
    best_init_functions = [
        get_initialization_function_by_enum(InitializationFunctions.XAVIER),
    ]

    @staticmethod
    def activation_function(x):
        return np.maximum(x, np.zeros(x.shape))

    @staticmethod
    def derivation_of_activation_function(x):
        return np.vectorize(lambda x: 0 if x < 0 else 1)(x)


class CustomActivationFuncHasNoInitializationFuncError(ValueError):
    message = (
        "When you use own activation function "
        "you must setup the initialization function by your self"
    )

    def __init__(self, message=None):
        self.message = message or self.message
        super().__init__(self.message)


class Custom(ABCActivationFunction):
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
    ActivationFunctions.SIGMOID: Sigmoid,
    ActivationFunctions.RELU: ReLu,
}


def get_activation_function_by_enum(enum_val: InitializationFunctions):
    return activation_function_by_enum[enum_val]()


def Softmax_grad(x):  # Best implementation (VERY FAST)
    """Returns the jacobian of the Softmax function for the given set of inputs.
    Inputs:
    x: should be a 2d array where the rows correspond to the samples
        and the columns correspond to the nodes.
    Returns: jacobian
    """
    s = Softmax(x)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
    temp1 = np.einsum("ij,jk->ijk", s, a)
    temp2 = np.einsum("ij,ik->ijk", s, s)
    return temp1 - temp2


def Softmax(x):
    """
    Performs the softmax activation on a given set of inputs
    Input: x (N,k) ndarray (N: no. of samples, k: no. of nodes)
    Returns:
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    """
    max_x = np.amax(x, 1).reshape(x.shape[0], 1)  # Get the row-wise maximum
    e_x = np.exp(x - max_x)  # For stability
    return e_x / e_x.sum(axis=1, keepdims=True)
