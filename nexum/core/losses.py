from abc import ABC, abstractstaticmethod
from enum import Enum

import numpy as np

from nexum.services.enums import ContainsEnumMeta


class Losses(str, Enum, metaclass=ContainsEnumMeta):
    MSE = "mse"
    BCE = "bce"


class ABCLoss(ABC):
    @abstractstaticmethod
    def calculate(real, predicted):
        raise NotImplementedError()

    @abstractstaticmethod
    def derivation(real, predicted):
        raise NotImplementedError()

    def __call__(self, real, predicted):
        return self.calculate(real, predicted)


class MSE(ABCLoss):
    @staticmethod
    def calculate(real, predicted):
        return np.mean(np.power(real - predicted, 2))

    @staticmethod
    def derivation(real, predicted):
        return 2 * (predicted - real) / np.size(real)


class BinaryCrossEntropy(ABCLoss):
    @staticmethod
    def calculate(real, predicted):
        return np.mean(-real * np.log(predicted) - (1 - real) * np.log(1 - predicted))

    @staticmethod
    def derivation(real, predicted):
        return ((1 - real) / (1 - predicted) - real / predicted) / np.size(real)


loss_by_enum = {
    Losses.MSE: MSE,
    Losses.BCE: BinaryCrossEntropy,
}


def get_loss_by_enum(enum_value: Losses):
    return loss_by_enum[enum_value]()
