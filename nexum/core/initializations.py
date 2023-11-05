from enum import Enum

import numpy as np

from nexum.services.enums import ContainsEnumMeta


class InitializationFunctions(str, Enum, metaclass=ContainsEnumMeta):
    RANDOM_1 = "random_from_0_to_1"
    RANDOM_2 = "random_from_-1_to_1"
    XAVIER = "xavier"


def xavier_init(input_units, output_units):
    variance = 1 / (input_units + output_units)
    std_dev = np.sqrt(variance)
    weights = np.random.normal(loc=0.0, scale=std_dev, size=(input_units, output_units))
    return weights


def get_random_init_function_from_a_to_b(a, b):
    def rand_function(out_n, in_n):
        length = b - a
        return np.random.rand(out_n, in_n) * length - a

    return rand_function


initialization_function_by_enum = {
    InitializationFunctions.RANDOM_1: get_random_init_function_from_a_to_b(0, 1),
    InitializationFunctions.RANDOM_2: get_random_init_function_from_a_to_b(-1, 1),
    InitializationFunctions.XAVIER: xavier_init,
}


def get_initialization_function_by_enum(enum_val: InitializationFunctions):
    return initialization_function_by_enum[enum_val]
