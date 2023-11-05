from enum import Enum

import numpy as np

from nexum.core.losses import ABCLoss, get_loss_by_enum, Losses
from nexum.core.layers import (
    BaseLayer,
    ConnectionTypes,
    FullConnectedLayer,
    InputLayer,
    OutputLayer,
)
from nexum.services.enums import EnumMeta
from nexum.services.iteration_logger import IterationLogger
from nexum.services.utils import accuracy_score


class WrongLayerTypeError(TypeError):
    ...


class LoggingEnum(Enum, metaclass=EnumMeta):
    ALL = "all"
    EPOCHS = "epochs"
    OFF = "off"


class SampleLogger(IterationLogger):
    modules_separator = " - "

    def __init__(self):
        modules = {
            "took_time": self.took_time,
            # "accuracy": self.accuracy,
            "error": self.error,
        }

        super().__init__(modules=modules)

    @staticmethod
    def error(error):
        return f"error: {error:.5f}"

    @staticmethod
    def accuracy(accuracy):
        return f"accuracy: {accuracy:.5f}"

    def set_sample_n(self, n: int) -> None:
        self.desc = f"Sample {n}: "


class EpochLogger(IterationLogger):
    desc = "Epochs: "
    modules_separator = " - "

    def __init__(self):
        modules = {
            "error": self.error,
            "accuracy": self.accuracy,
        }

        super().__init__(modules=modules)

    @staticmethod
    def error(error):
        return f"error: {error:.5f}"

    @staticmethod
    def accuracy(accuracy):
        return f"accuracy: {accuracy:.5f}"


class Perceptron:
    def __init__(
        self,
        layers_config: list[int | BaseLayer],
        logging=LoggingEnum.EPOCHS,
        loss=Losses.MSE,
    ):
        self.layers: list[int | BaseLayer]
        self.sample_logger = SampleLogger()
        self.epoch_logger = EpochLogger()
        self._init_layers(layers_config)
        self._init_loss(loss)
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

    @property
    def w(self):
        w = []

        for i in range(len(self.layers) - 1):
            w.append(self.layers[i + 1].w)

        return w

    def _init_loss(self, loss: Losses):
        self.loss: ABCLoss = get_loss_by_enum(loss)

    def _init_layers(self, config: list[int | BaseLayer]):
        self.layers = []

        def get_layer_cls(*args, **kwargs):
            return OutputLayer(
                *args, connection_type=ConnectionTypes.FULL_CONNECTED, **kwargs
            )

        for i, item in enumerate(config):
            if i == 0:
                layer_cls = InputLayer
            elif i == len(config) - 1:
                layer_cls = get_layer_cls
            else:
                layer_cls = FullConnectedLayer

            if isinstance(item, int):
                layer = layer_cls(item)
            elif isinstance(item, BaseLayer):
                layer = item
            else:
                raise WrongLayerTypeError(
                    f"Layer {i} has wrong type must be int or Layer type."
                )

            self.layers.append(layer)

            if i == 0:
                continue

            layer.connect_to_layer(self.layers[i - 1])

    @property
    def save_data(self):
        return self._save_data

    @save_data.setter
    def save_data(self, value: bool):
        self._save_data = value
        for layer in self.layers:
            layer.save_data = value

    def finalize(self, values):
        return values

    def predict(self, value, train=False, finalize=True):
        if not train:
            value = np.reshape(value, (*value.shape, 1))

        result_data = value
        for layer in self.layers:
            result_data = layer.calculate(result_data)

        if finalize:
            result_data = self.finalize(result_data)

        if not train:
            return np.reshape(result_data, result_data.shape[:-1])

        return result_data

    def log_training_progress(self, i, targets, results, training_data):
        self.sample_logger.ds.accuracy = accuracy_score(
            targets[: i + 1], results[: i + 1]
        )

    def back_propagation_iteration(self, results, target, learning_rate):
        layers_n = len(self.layers)

        for i in range(layers_n - 1, 0, -1):
            current_layer = self.layers[i]

            if i == layers_n - 1:
                current_layer.deltas = results - target
                current_layer.deltas *= current_layer.train_function(results)

            else:
                next_layer = self.layers[i + 1]
                deltas = np.empty(current_layer.node_number)

                for j in range(current_layer.node_number):
                    deltas[j] = np.sum(
                        next_layer.w[:, j]
                        * next_layer.deltas
                        * current_layer.train_function(next_layer.nodes)
                    )

                current_layer.deltas = deltas

        for i in range(1, layers_n):
            current_layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            for j in range(current_layer.node_number):
                current_layer.w[j] -= (
                    learning_rate * current_layer.deltas[j] * prev_layer.nodes
                )

            current_layer.bias -= learning_rate * current_layer.deltas

    def back_propagation(self, training_data, targets, learning_rate, epochs):
        training_data = training_data.copy()
        targets = targets.copy()

        self.save_data = True

        epoch_range = range(epochs)
        if self.logging in (LoggingEnum.ALL, LoggingEnum.EPOCHS):
            epoch_range = self.epoch_logger(epoch_range, position=0)

        for epoch in epoch_range:
            # self.sample_logger.desc = f"Sample {epoch+1}: "

            randomize = np.arange(len(training_data))
            np.random.shuffle(randomize)
            training_data = training_data[randomize]
            targets = targets[randomize]
            results = np.empty(targets.shape)

            sample_range = range(training_data.shape[0])
            if self.logging == LoggingEnum.ALL:
                sample_range = self.sample_logger(sample_range, position=1)

            for i in sample_range:
                result = self.predict(training_data[i], finalize=False)
                results[i] = result
                self.back_propagation_iteration(result, targets[i], learning_rate)

                self.log_training_progress(i, targets, results, training_data)

        self.save_data = False

    def new_trainer(self, training_data, targets, learning_rate, epochs):
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
                output = self.predict(x, train=True, finalize=False)

                # error
                error += self.loss(y, output)

                # backward
                grad = self.loss.derivation(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

                # add data to sample_logger to display current training parameters
                self.sample_logger.ds.error = error / (i + 1)

            # add data to epoch_logger to display current training parameters
            self.epoch_logger.ds.error = error / training_data.shape[0]
            self.epoch_logger.ds.accuracy = accuracy_score(
                targets, self.predict(training_data, train=True)
            )

    def train(
        self,
        training_data,
        targets,
        learning_rate=0.01,
        epochs=100,
        algorithm="back_propagation",
    ):
        match algorithm:
            case "back_propagation":
                self.new_trainer(training_data, targets, learning_rate, epochs)
            case _:
                self.back_propagation(training_data, targets, learning_rate, epochs)
