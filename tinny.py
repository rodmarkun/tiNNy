import utils
import numpy as np
from activation_functions import functions, derivates

class TiNNyNetwork:
    def __init__(self, layers=[]):
        self.layers = layers
    
    def train(self, X_train, y_train, iterations, learning_rate=0.001):
        for i in range(iterations):
            input = X_train
            for layer in self.layers:
                layer.forward(input)
                input = layer.output
            reversed_layers = self.layers[::-1]
            for idx, layer in enumerate(reversed_layers):
                if isinstance(layer, DenseLayer):
                    next_layer = reversed_layers[idx-1]
                    layer.backward(X_train.shape[0], next_layer.weights, next_layer.d_value)
                elif isinstance(layer, OutputLayer):
                    layer.backward(X_train.shape[0], y_train)
            for layer in self.layers:
                layer.update_parameters(learning_rate)

            if i % 10 == 0:
                print(f"Iteration {i}")
                predictions = utils.get_predictions(self.layers[-1].output)
                print(f"Accuracy: {utils.get_accuracy(predictions, y_train)}")

    def test(self, X_test, y_test):
        input = X_test
        for layer in self.layers:
            layer.forward(input)
            input = layer.output
        predictions = utils.get_predictions(self.layers[-1].output)
        print(f"Accuracy in test phase: {utils.get_accuracy(predictions, y_test)}")


class Layer:
    def __init__(self, number_inputs, number_neurons, activation) -> None:
        self.weights = np.random.rand(number_neurons, number_inputs) - 0.5
        self.biases = np.random.rand(number_neurons, 1) - 0.5
        self.activation_function = functions[activation]
        self.derivate_function = derivates[activation]
        self.d_value = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, X):
        self.input = X
        self.value = self.weights.dot(X) + self.biases
        self.output = self.activation_function(self.value)

    def update_parameters(self, learning_rate):
        self.weights = self.weights - learning_rate * self.d_weights
        self.biases = self.biases - learning_rate * self.d_biases

    def display_info(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}\n-------")

class OutputLayer(Layer):
    def __init__(self, number_inputs, number_neurons, activation) -> None:
        super().__init__(number_inputs, number_neurons, activation)
    
    def backward(self, number_instances, y):
        y = utils.one_hot(y)
        self.d_value = self.output - y
        self.d_weights = 1 / number_instances * self.d_value.dot(self.input.T)
        self.d_biases = 1 / number_instances * np.sum(self.d_value)

class DenseLayer(Layer):
    def __init__(self, number_inputs, number_neurons, activation) -> None:
        super().__init__(number_inputs, number_neurons, activation)
    
    def backward(self, number_instances, weights_of_next_layer, d_value_of_next_layer):
        self.d_value = weights_of_next_layer.T.dot(d_value_of_next_layer) * self.derivate_function(self.value)
        self.d_weights = 1 / number_instances * self.d_value.dot(self.input.T)
        self.d_biases = 1 / number_instances * np.sum(self.d_value)