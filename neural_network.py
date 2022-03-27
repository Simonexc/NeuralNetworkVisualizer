from typing import List, Callable, Tuple

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, input_size: int, output_size: int, layers_size: List[int], activation_func: Callable):
        self.input_size = input_size
        self.output_size = output_size
        self.layers_size = layers_size
        self.activation_func = activation_func

        self.weights, self.biases = self._generate_structure(layers_size)

    def _generate_structure(self, layers_size: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        weights = []
        biases = []

        previous_layer_size = self.input_size
        for layer_size in layers_size:
            weights.append(np.array([np.random.rand(previous_layer_size) for _ in range(layer_size)]))
            biases.append(np.random.rand(layer_size))
            previous_layer_size = layer_size

        weights.append([np.random.rand(previous_layer_size) for _ in range(self.output_size)])

        return np.array(weights), np.array(biases)

    @staticmethod
    def perceptron_output(input_tensor: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        return np.dot(input_tensor, weights.reshape(weights.shape[0], 1)).reshape(-1) + bias
