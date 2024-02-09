import numpy as np

# Definition of problem types (regression and classification)
PROBLEM_TYPES = {"R": "regression", "C": "classification"}

def one_hot(y: np.array):
    """    
    Converts a vector of labels into a one-hot encoded matrix.

    Args:
        y (np.array): A vector of integer labels ranging from 0 to n_classes - 1.

    Returns:
        np.array: A matrix of shape (n_classes, n_samples) where each column is a one-hot encoded representation of the corresponding label in y.
    """

    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y.astype(int)] = 1
    return one_hot.T

def get_accuracy(predictions: np.array, y: np.array):
    """
    Calculates the accuracy of predictions against the true labels.

    Args:
        predictions (np.array): An array of predictions.
        y (np.array): An array of true labels.

    Returns:
        float: Accuracy of the predictions, defined as the proportion of correct predictions over the total number of predictions. 
    """

    return np.sum(predictions == y) / y.size

def get_predictions(y: np.array):
    """
    Converts a matrix of class probabilities into class predictions by selecting the class with the highest probability for each sample.

    Returns:
        np.array: A vector of predictions, where each element is the predicted class for the corresponding sample.
    """
    
    return np.argmax(y, 0)