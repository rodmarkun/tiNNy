import utils
import numpy as np
import activation_functions as af
import loss_functions as lf

class TiNNyNetwork:
    """
    A minimal neural network class designed to handle basic machine learning tasks.
    """

    def __init__(self, problem_type: str, loss_function: str, layers: list = [], iteration_step: int = 10):
        """
        Creates a new TiNNy neural network.

        Args:
            problem_type (str): Type of the problem the network is meant to solve. Can be either 'classification' or 'regression'.
            loss_function (str): Function that will compute the loss of the neutal network in each iteration. Loss functions can be found in 'loss_functions.py'.
            layers (list, optional): Neuron layers that will compose the neural network. Defaults to [].
            iteration_step (int, optional): The step size for printing iteration-specific information during training. Defaults to 10.

        Raises:
            Exception: If problem_type is not valid, an Exception will be thrown as the network will not know how to compute certain values.
        """

        if problem_type.lower() not in utils.PROBLEM_TYPES.values():
            raise Exception(f"Problem type not specified or incorrectly spelled. Only possible values are: {utils.PROBLEM_TYPES.values()}")
        self.problem_type = problem_type.lower()
        self.loss_function = lf.functions[loss_function]
        self.layers = layers
        self.output_layer = self.layers[-1]
        self.output_layer.problem_type = self.problem_type
        self.iteration_step = iteration_step

    def make_prediction(self, X: np.array):
        """
        Makes a prediction based on the input X through the network.

        Args:
            X (np.array): Input data in np.array form for which predictions are to be made.
        """

        input = X
        for layer in self.layers:
            layer.forward(input)
            input = layer.output
    
    def train(self, X_train: np.array, y_train: np.array, iterations: int, learning_rate: float = 0.01):
        """
        Trains the network on the provided training data and labels for a specified number of iterations and learning rate.

        Args:
            X_train (np.array): Training set of features in np.array form.
            y_train (np.array): Training set of labels in np.array form.
            iterations (int): Number of iterations the network will run to train itself.
            learning_rate (float, optional): Learning rate for parameter updates. Defaults to 0.01.
        """

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
                prediction = self.output_layer.get_prediction()
                loss = self.loss_function(y_train, prediction)
                print(f"Loss: {loss}")
                if self.problem_type == utils.PROBLEM_TYPES["C"]:
                    print(f"Accuracy: {utils.get_accuracy(prediction, y_train)}")

    def test(self, X_test: np.array, y_test: np.array, plot: bool = False):
        """
        Tests the network on the provided test data and labels.

        Args:
            X_test (np.array): Testing set of features in np.array form.
            y_test (np.array): Testing set of labels in np.array form.
            plot (bool, optional): Generates a scatter plot of true vs predicted values. Defaults to False.
        """

        self.make_prediction(X_test)
        prediction = self.output_layer.get_prediction()
        loss = self.loss_function(y_test, prediction)
        print(f"Loss in test: {loss}")
        if self.problem_type == utils.PROBLEM_TYPES["C"]:
            print(f"Accuracy in test: {utils.get_accuracy(prediction, y_test)}")
        # Predicted values vs true values visualization
        if plot:
            if self.problem_type == utils.PROBLEM_TYPES["R"]:
                utils.regression_scatter_plot(y_test, prediction)
            elif self.problem_type == utils.PROBLEM_TYPES["C"]:
                utils.classification_scatter_plot(y_test, prediction, np.unique(np.concatenate((y_test, prediction))))




class Layer:
    """
    Parent class for all layers that can be added onto the network.
    """

    def __init__(self, number_inputs: int, number_neurons: int, activation: str) -> None:
        """
        Creates a new generic layer.

        Args:
            number_inputs (int): Number of input features the layer will have.
            number_neurons (int): Number of neurons present in this layer.
            activation (str): Activation function the neurons in this layer make use of. Activation functions can be found in 'activation_functions.py'
        """

        self.weights = np.random.rand(number_neurons, number_inputs) - 0.5
        self.biases = np.random.rand(number_neurons, 1) - 0.5
        self.activation_function = af.functions[activation]
        self.derivate_function = af.derivates[activation]
        self.d_value = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, X: np.array):
        """
        Computes the forward pass through the layer using input X.

        Args:
            X (np.array): Input data in np.array form for which predictions are to be made.
        """
        
        self.input = X
        self.value = self.weights.dot(X) + self.biases
        self.output = self.activation_function(self.value)

    def update_parameters(self, learning_rate: int):
        """
        Updates the layer's parameters (weights and biases) based on their gradients and the learning rate.

        Args:
            learning_rate (int): Learning rate for parameter updates.
        """

        self.weights = self.weights - learning_rate * self.d_weights
        self.biases = self.biases - learning_rate * self.d_biases

    def display_info(self):
        """
        Prints the layer's attributes for debugging purposes.
        """

        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}\n-------")

class OutputLayer(Layer):
    """
    Represents the output layer of a neural network, extending the basic layer functionality.
    """

    def __init__(self, number_inputs: int, number_neurons: int, activation: str) -> None:
        """
        Creates a new TiNNy Output layer.

        Args:
            number_inputs (int): Number of input features the layer will have.
            number_neurons (int): Number of neurons present in this layer.
            activation (str): Activation function the neurons in this layer make use of. Activation functions can be found in 'activation_functions.py'
        """

        super().__init__(number_inputs, number_neurons, activation)
        self.problem_type = None
    
    def backward(self, number_instances: int, y: np.array):
        """
        Computes the backward pass specifically for the output layer, updating gradients based on the loss.

        Args:
            number_instances (int): Number of instances used in the forward pass.
            y (np.array): True values (labels) in np.array form for the instances used in the forward pass. 
        """

        if self.problem_type == utils.PROBLEM_TYPES["C"]:
            y = utils.one_hot(y)
        self.d_value = self.output - y
        self.d_weights = 1 / number_instances * self.d_value.dot(self.input.T)
        self.d_biases = 1 / number_instances * np.sum(self.d_value)

    def get_prediction(self):
        """
        Returns predictions from the output layer's activations, adjusted for the problem type.

        Returns:
            np.array: Predictions made for input X in np.array form.
        """

        if self.problem_type == utils.PROBLEM_TYPES["C"]:
            return np.argmax(self.output, 0)
        else:
            return self.output

class DenseLayer(Layer):
    """
    Represents a dense (fully connected) layer in a neural network, extending the basic layer functionality.
    """

    def __init__(self, number_inputs: int, number_neurons: int, activation: str) -> None:
        """
        Creates a new TiNNy Dense layer.

        Args:
            number_inputs (int): Number of input features the layer will have.
            number_neurons (int): Number of neurons present in this layer.
            activation (str): Activation function the neurons in this layer make use of. Activation functions can be found in 'activation_functions.py'
        """

        super().__init__(number_inputs, number_neurons, activation)
    
    def backward(self, number_instances: int, weights_of_next_layer: np.array, d_value_of_next_layer: np.array):
        """
        Computes the backward pass for the dense layer, updating gradients based on subsequent layer's gradients.

        Args:
            number_instances (int): Number of instances used in the forward pass.
            weights_of_next_layer (np.array): Weights of the neurons in the next layer. In np.array form.
            d_value_of_next_layer (np.array): Gradient of the value acquired in the next layer. In np.array form.
        """

        self.d_value = weights_of_next_layer.T.dot(d_value_of_next_layer) * self.derivate_function(self.value)
        self.d_weights = 1 / number_instances * self.d_value.dot(self.input.T)
        self.d_biases = 1 / number_instances * np.sum(self.d_value)