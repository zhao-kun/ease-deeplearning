import numpy as np

from layer import Layer

class Dense(Layer):
    def __init__(self, input_size :int, output_size :int):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input :np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient :np.ndarray, learning_rate :float) -> np.ndarray:
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.biases -= learning_rate * output_gradient
        self.weights -= learning_rate * weights_gradient
        return np.dot(self.weights.T, output_gradient)