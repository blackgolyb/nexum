from typing import Self

import numpy as np


class Layer:
    def __init__(self, node_number, bias=1):
        self.bias = bias
        self.is_first_layer = True
        self.is_last_layer = False
        self.node_number = node_number
        self.nodes = [i for i in range(node_number)]
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        # self.activation_function = lambda x: x >= 0.5

    @property
    def node_number(self):
        return self._node_number + 1 if self.bias is not None else 0

    @node_number.setter
    def node_number(self, node_number):
        self._node_number = node_number

    def set_input_values(self, values: np.array):
        if self.bias is not None:
            self._nodes = np.append(values, self.bias)
        else:
            self._nodes = values

    def connect_layer(self, layer: Self):
        self.is_first_layer = False
        self.parent_layer = layer

        self.w = np.random.rand(self._node_number, layer.node_number) * 2 - 1
        # self.w = np.ones((self._node_number, layer.node_number))

    def node_calculate_function(self, values: np.array, node_id):
        return self.activation_function(np.sum(values * self.w[node_id]))

    def calculate(self):
        if self.is_first_layer:
            return self._nodes

        input_nodes = self.parent_layer.calculate()

        nodes = []
        for i in range(self._node_number):
            nodes.append(self.node_calculate_function(input_nodes, i))

        if self.bias is not None and not self.is_last_layer:
            nodes.append(self.bias)

        return np.array(nodes)
