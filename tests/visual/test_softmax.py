import numpy as np
from sklearn.datasets import load_iris

from nexum.core.layers import OutputLayer
from nexum.core.models import Perceptron
from nexum.services.utils import accuracy_score


def one_hot(indices):
    unique_values = list(np.unique(indices))
    depth = len(unique_values)
    results = np.zeros((indices.shape[0], depth))

    for i in range(indices.shape[0]):
        index = unique_values.index(indices[i])
        results[i][index] = 1

    return results


def get_training_data(x, y, training_size: float):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)

    training_x = x[randomize]
    training_y = y[randomize]

    training_n = int(training_size * len(x))

    return training_x[: training_n + 1], training_y[: training_n + 1]


def display_compare_table(input_data, targets, nn):
    yes = "✅"
    no = "❌"
    row_table = [
        ["data", "target", "result", "assert"],
    ]
    results = []

    for i in range(input_data.shape[0]):
        result = nn.predict(input_data[i])
        results.append(result)
        assert_result = yes if np.array_equal(targets[i], result) else no
        row_table.append([input_data[i], targets[i], result, assert_result])

    print("accuracy: ", accuracy_score(targets, results))

    print(
        "\n".join(list(map(lambda x: "\t".join(map(lambda y: str(y), x)), row_table)))
    )


step_function = np.vectorize(lambda x: float(x >= 0.5))


def test_softmax():
    iris = load_iris()

    training_size = 0.3
    epochs = 1000
    learning_rate = 0.08

    X = iris["data"]
    y = iris["target"]
    y = one_hot(y)

    t_x, t_y = get_training_data(X, y, training_size)

    o_l = OutputLayer(3, activation_function="softmax")
    nn = Perceptron(layers_config=[4, 5, o_l])
    nn.finalize = step_function

    nn.train(t_x, t_y, epochs=epochs, learning_rate=learning_rate)

    display_compare_table(X, y, nn)
