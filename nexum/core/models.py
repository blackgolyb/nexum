import numpy as np

from nexum.core.layers import (
    BaseLayer,
    ConnectionTypes,
    Dense,
    InputLayer,
    OutputLayer,
)
from nexum.core.losses import ABCLoss, Losses, get_loss_by_enum
from nexum.core.trainer import GradientTrainer, LoggingEnum


class WrongLayerTypeError(TypeError):
    ...


class Sequential:
    def __init__(
        self,
        layers_config: list[int | BaseLayer],
        logging=LoggingEnum.EPOCHS,
        loss=Losses.MSE,
    ):
        self.layers: list[int | BaseLayer]
        self._init_layers(layers_config)
        self._init_loss(loss)
        self.trainer = GradientTrainer()
        self.logging = logging

    @property
    def logging(self) -> LoggingEnum:
        return self.trainer.logging

    @logging.setter
    def logging(self, value: LoggingEnum) -> None:
        self.trainer.logging = value

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

        for i, item in enumerate(config):
            if not isinstance(item, BaseLayer):
                raise WrongLayerTypeError(
                    f"Layer {i} has wrong type must be Layer type."
                )

            self.layers.append(item)

            if i == 0:
                continue

            item.connect_to_layer(self.layers[i - 1])

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

    def train(self, training_data, targets, learning_rate=0.01, epochs=100):
        self.trainer.train(
            training_data, targets, learning_rate, epochs, nn=self, loss=self.loss
        )


class Perceptron(Sequential):
    def _init_layers(self, config: list[int | BaseLayer]):
        self.layers = []

        def get_layer_cls(*args, **kwargs):
            return OutputLayer(*args, connection_type=ConnectionTypes.DENSE, **kwargs)

        for i, item in enumerate(config):
            if i == 0:
                layer_cls = InputLayer
            elif i == len(config) - 1:
                layer_cls = get_layer_cls
            else:
                layer_cls = Dense

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
