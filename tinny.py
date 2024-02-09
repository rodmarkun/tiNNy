import utils
import numpy as np
import activation_functions as af
import loss_functions as lf

class TiNNyNetwork:
    def __init__(self, problem_type, loss_function, layers=[], iteration_step=10):
        if problem_type.lower() not in utils.PROBLEM_TYPES.values():
            raise Exception(f"Problem type not specified or incorrectly spelled. Only possible values are: {utils.PROBLEM_TYPES.values()}")
        self.problem_type = problem_type.lower()
        self.loss_function = lf.functions[loss_function]
        self.layers = layers
        self.output_layer = self.layers[-1]
        self.output_layer.problem_type = self.problem_type
        self.iteration_step = iteration_step

    def make_prediction(self, X):
        input = X
        for layer in self.layers:
            layer.forward(input)
            input = layer.output
    
    def train(self, X_train, y_train, iterations, learning_rate=0.01):
        for i in range(iterations):
            self.make_prediction(X_train)
            reversed_layers = self.layers[::-1]
            for idx, layer in enumerate(reversed_layers):
                if isinstance(layer, DenseLayer):
                    next_layer = reversed_layers[idx-1]
                    layer.backward(X_train.shape[0], next_layer.weights, next_layer.d_value)
                elif isinstance(layer, OutputLayer):
                    layer.backward(X_train.shape[0], y_train)
            for layer in self.layers:
                layer.update_parameters(learning_rate)

            if i % self.iteration_step == 0:
                print(f"Iteration {i}")
                prediction = self.output_layer.get_prediction(self.problem_type)
                loss = self.loss_function(y_train, prediction)
                print(f"Loss: {loss}")
                if self.problem_type == utils.PROBLEM_TYPES["C"]:
                    print(f"Accuracy: {utils.get_accuracy(prediction, y_train)}")

    def test(self, X_test, y_test):
        self.make_prediction(X_test)
        prediction = self.output_layer.get_prediction(self.problem_type)
        loss = self.loss_function(y_test, prediction)
        print(f"Loss in test: {loss}")
        if self.problem_type == utils.PROBLEM_TYPES["C"]:
            print(f"Accuracy in test: {utils.get_accuracy(prediction, y_test)}")


class Layer:
    def __init__(self, number_inputs, number_neurons, activation) -> None:
        self.weights = np.random.rand(number_neurons, number_inputs) - 0.5
        self.biases = np.random.rand(number_neurons, 1) - 0.5
        self.activation_function = af.functions[activation]
        self.derivate_function = af.derivates[activation]
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
        self.problem_type = None
    
    def backward(self, number_instances, y):
        if self.problem_type == utils.PROBLEM_TYPES["C"]:
            y = utils.one_hot(y)
        self.d_value = self.output - y
        self.d_weights = 1 / number_instances * self.d_value.dot(self.input.T)
        self.d_biases = 1 / number_instances * np.sum(self.d_value)

    def get_prediction(self, problem_type):
        if problem_type == utils.PROBLEM_TYPES["C"]:
            return np.argmax(self.output, 0)
        else:
            return self.output

class DenseLayer(Layer):
    def __init__(self, number_inputs, number_neurons, activation) -> None:
        super().__init__(number_inputs, number_neurons, activation)
    
    def backward(self, number_instances, weights_of_next_layer, d_value_of_next_layer):
        self.d_value = weights_of_next_layer.T.dot(d_value_of_next_layer) * self.derivate_function(self.value)
        self.d_weights = 1 / number_instances * self.d_value.dot(self.input.T)
        self.d_biases = 1 / number_instances * np.sum(self.d_value)