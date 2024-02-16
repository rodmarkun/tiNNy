import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

def regression_scatter_plot(y_test, prediction):
    plt.scatter(y_test, prediction)
    plt.title('Predicted vs. Real Values')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    max_value = np.max([np.max(y_test), np.max(prediction)])
    min_value = np.min([np.min(y_test), np.min(prediction)])
    plt.plot([min_value, max_value], [min_value, max_value], color='red') 
    plt.show()

def classification_scatter_plot(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

