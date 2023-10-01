import numpy as np

from services.layer import Layer


class WrongLayerTypeError(TypeError):
    ...


class NeuralNetwork:
    def __init__(self, layers_config: list[int | Layer]):
        self.layers: list[int | Layer]
        self._init_layers(layers_config)
        ...

    @property
    def w(self):
        w = []

        for i in range(len(self.layers) - 1):
            w.append(self.layers[i + 1].w)

        return w

    def _init_layers(self, config: list[int | Layer]):
        self.layers = []

        for i, item in enumerate(config):
            if isinstance(item, int):
                layer = Layer(item)
            elif isinstance(item, Layer):
                layer = item
            else:
                raise WrongLayerTypeError(
                    f"Layer {i} has wrong type must be int or Layer type."
                )

            self.layers.append(layer)

            if i == 0:
                continue

            layer.connect_layer(self.layers[i - 1])

        self.layers[-1].is_last_layer = True

    def feat_forward(self, value):
        self.layers[0].set_input_values(value)
        return self.layers[-1].calculate()
