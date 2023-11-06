from abc import ABC
from enum import Enum

import numpy as np

from nexum.services.enums import EnumMeta
from nexum.services.iteration_logger import EpochLogger, SampleLogger
from nexum.services.utils import accuracy_score


class LoggingEnum(Enum, metaclass=EnumMeta):
    ALL = "all"
    EPOCHS = "epochs"
    OFF = "off"


class ABCTrainer(ABC):
    def __init__(self, logging=LoggingEnum.EPOCHS):
        self.sample_logger = SampleLogger()
        self.epoch_logger = EpochLogger()
        self.logging = logging

    @property
    def logging(self) -> LoggingEnum:
        return self._logging

    @logging.setter
    def logging(self, value: LoggingEnum) -> None:
        self._logging = value

        if self._logging == LoggingEnum.EPOCHS:
            self.epoch_logger.logging = True
            self.sample_logger.logging = False
        elif self.logging == LoggingEnum.ALL:
            self.epoch_logger.logging = True
            self.sample_logger.logging = True
        else:
            self.epoch_logger.logging = False
            self.sample_logger.logging = False

    def train(self):
        ...


class GradientTrainer(ABCTrainer):
    def train(
        self,
        training_data,
        targets,
        learning_rate,
        epochs,
        *,
        nn,
        loss,
    ):
        training_data = np.reshape(training_data, (*training_data.shape, 1))
        targets = np.reshape(targets, (*targets.shape, 1))

        epoch_range = self.epoch_logger(range(epochs), position=0)

        for epoch in epoch_range:
            error = 0

            sample_range = self.sample_logger(range(training_data.shape[0]), position=1)
            self.sample_logger.set_sample_n(epoch + 1)

            for i in sample_range:
                # data
                x = training_data[i]
                y = targets[i]

                # forward
                output = nn.predict(x, train=True, finalize=False)

                # error
                error += loss(y, output)

                # backward
                grad = loss.derivation(y, output)
                for layer in reversed(nn.layers):
                    grad = layer.backward(grad, learning_rate)

                # add data to sample_logger to display current training parameters
                self.sample_logger.ds.error = error / (i + 1)

            # add data to epoch_logger to display current training parameters
            predicted_data = np.empty(targets.shape)
            for i in range(targets.shape[0]):
                predicted_data[i] = nn.predict(training_data[i], train=True)

            self.epoch_logger.ds.error = error / training_data.shape[0]
            self.epoch_logger.ds.accuracy = accuracy_score(targets, predicted_data)
