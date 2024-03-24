import numpy as np
import utils

def crossentropy(y_true: np.array, prediction: np.array):
    """
    Computes the cross-entropy loss between true labels and predictions.
    This function first converts the true labels into a one-hot encoded format, then calculates the cross-entropy loss.

    Args:
        y_true (np.array): A vector/matrix of true labels.
        prediction (np.array): A vector/matrix of predicted probabilities for each class.

    Returns:
        float: The cross-entropy loss averaged over all samples.
    """

    y_true = utils.one_hot(y_true)
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(prediction + 1e-15)) / m
    return loss

def mse(y_true: np.array, prediction: np.array):
    """
    Computes the Mean Squared Error (MSE) loss between true values and predictions.

    Args:
        y_true (np.array): A vector/matrix of true target values.
        prediction (np.array): A vector/matrix of predictions.

    Returns:
        float: The MSE loss, calculated as the average of the squared differences between true values and predictions.
    """
    
    squared_diffs = np.square(y_true - prediction)
    mse_loss = np.mean(squared_diffs)
    return mse_loss

functions = {
    "crossentropy": crossentropy,
    "mse": mse
}