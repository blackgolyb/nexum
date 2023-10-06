import sys
import time

from nexum.core.layers import (
    BaseLayer,
    InputLayer,
    OutputLayer,
    FullConnectedLayer,
    ConnectionTypes,
)
from nexum.services.utils import accuracy_score

import numpy as np


class IterationLogger(object):
    progress_bar_len = 30
    modules_separator = " "
    log_long_in_long_format = False
    iteration_and_progress_bar_fmt = "{iterations} {bar}"

    def __init__(self, modules):
        self.modules = modules
        self.prev_modules_string = ""

    @staticmethod
    def iterations(iteration, n_iterations):
        return f"%{len(str(n_iterations))}d/{n_iterations}" % iteration

    @classmethod
    def progress_bar(cls, iteration, n_iterations, progress_bar_len=None):
        if progress_bar_len is None:
            progress_bar_len = cls.progress_bar_len

        bar_progress = int(iteration / n_iterations * progress_bar_len)

        bar_processed = "=" * bar_progress
        if bar_progress == 0:
            bar_processed = ""
        elif bar_progress != progress_bar_len:
            bar_processed = f"{bar_processed[:-1]}>"

        return f"[{bar_processed}{'.'*(progress_bar_len - bar_progress)}]"

    @classmethod
    def iteration_and_progress_bar(cls, iteration, n_iterations, **kwargs):
        iterations = cls.iterations(iteration, n_iterations)
        bar = cls.progress_bar(iteration, n_iterations, **kwargs)
        return cls.iteration_and_progress_bar_fmt.format(iterations=iterations, bar=bar)

    def log_iteration(self, **modules_data):
        modules = []
        for module_name in self.modules:
            module_data = modules_data.get(module_name, None)
            if module_data is not None:
                modules.append(self.modules[module_name](**module_data))

        modules_string = self.modules_separator.join(modules)

        if not self.log_long_in_long_format:
            addition_symbols = " " * max(
                len(self.prev_modules_string) - len(modules_string), 0
            )
            sys.stdout.write(f"\r{modules_string}{addition_symbols}")
        else:
            print(modules_string)

        self.prev_modules_string = modules_string


class WrongLayerTypeError(TypeError):
    ...


class NNIterationLogger(IterationLogger):
    modules_separator = " - "

    def __init__(self):
        modules = {
            "iterations_bar": self.iteration_and_progress_bar,
            "took_time": self.took_time,
            "accuracy": self.accuracy,
        }

        super().__init__(modules=modules)

    @staticmethod
    def took_time(iteration, n_iterations, took_time):
        s = int(took_time)
        us = int((took_time * 100) % 100)
        if iteration != n_iterations:
            return f"ETA: {s}s {us}us"
        else:
            return f"{s}s {us}us/sample"

    @staticmethod
    def accuracy(accuracy):
        return f"accuracy: {accuracy:.5f}"


class Perceptron:
    def __init__(self, layers_config: list[int | BaseLayer]):
        self.layers: list[int | BaseLayer]
        self.iteration_logger = NNIterationLogger()
        self._init_layers(layers_config)

    @property
    def w(self):
        w = []

        for i in range(len(self.layers) - 1):
            w.append(self.layers[i + 1].w)

        return w

    def _init_layers(self, config: list[int | BaseLayer]):
        self.layers = []

        for i, item in enumerate(config):
            if i == 0:
                layer_cls = InputLayer
            elif i == len(config) - 1:
                layer_cls = lambda *args, **kwargs: OutputLayer(
                    *args, connection_type=ConnectionTypes.FULL_CONNECTED, **kwargs
                )
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

    def _on_save_data_in_layers(self):
        for layer in self.layers:
            layer.save_data = True

    def _off_save_data_in_layers(self):
        for layer in self.layers:
            layer.save_data = False

    def predict(self, value):
        self.layers[0].setup_input(value)
        return self.layers[-1].calculate()

    def log_training_progress(
        self, iteration, n_iterations, took_time=0, accuracy=None
    ):
        if accuracy is not None:
            accuracy = {
                "accuracy": accuracy,
            }
        self.iteration_logger.log_iteration(
            iterations_bar={
                "iteration": iteration,
                "n_iterations": n_iterations,
            },
            took_time={
                "iteration": iteration,
                "n_iterations": n_iterations,
                "took_time": took_time,
            },
            accuracy=accuracy,
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
                    deltas[j] = np.sum(next_layer.w[:, j] * next_layer.deltas)
                    deltas[j] *= current_layer.train_function(next_layer.nodes)

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
        all_took_time = 0

        self.save_data = True

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            one_epoch_took_time = 0

            randomize = np.arange(len(training_data))
            np.random.shuffle(randomize)
            training_data = training_data[randomize]
            targets = targets[randomize]
            results = np.empty(targets.shape)

            for i, data in enumerate(training_data):
                start_time = time.time()

                result = self.predict(data)
                results[i] = result
                self.back_propagation_iteration(result, targets[i], learning_rate)

                took_time = time.time() - start_time
                one_epoch_took_time += took_time

                took_time = (
                    took_time if i != len(training_data) - 1 else one_epoch_took_time
                )

                accuracy = None
                if i == training_data.shape[0] - 1:
                    accuracy = accuracy_score(targets, results)

                self.log_training_progress(
                    i + 1, len(training_data), took_time=took_time, accuracy=accuracy
                )

            all_took_time += one_epoch_took_time
            print("\n")

        self.save_data = False

        print(f"Training took: {all_took_time:.2f}s")

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
                self.back_propagation(training_data, targets, learning_rate, epochs)
            case _:
                self.back_propagation(training_data, targets, learning_rate, epochs)
