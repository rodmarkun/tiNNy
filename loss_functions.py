import numpy as np
import utils

def crossentropy(y_true, prediction):
    y_true = utils.one_hot(y_true)
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(prediction + 1e-15)) / m
    return loss

def mse(y_true, prediction):
    m = y_true.shape[1]
    loss = np.sum((prediction - y_true) ** 2) / m
    return loss

functions = {
    "crossentropy": crossentropy,
    "mse": mse
}