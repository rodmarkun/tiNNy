import numpy as np

PROBLEM_TYPES = {"R": "regression", "C": "classification"}

def one_hot(y):
    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y.astype(int)] = 1
    return one_hot.T

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def get_predictions(y):
    return np.argmax(y, 0)