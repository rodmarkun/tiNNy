import numpy as np

def ReLU(x: np.array):
    """
    Computes the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (np.array): Input array or matrix.

    Returns:
        np.array: An array where each element is the result of applying ReLU to the corresponding element of x, defined as max(x, 0).
    """

    return np.maximum(x, 0)

def ReLU_derivative(x: np.array):
    """
    Computes the derivative of the ReLU function.

    Args:
        x (np.array): Input array or matrix.

    Returns:
        _type_: An array where the derivative is 1 for all elements of x > 0, and 0 otherwise.
    """

    return (x > 0).astype(int)

def softmax(x: np.array):
    """
    Computes the softmax function for each column of the input matrix.

    Args:
        x (np.array): Input array or matrix.

    Returns:
        np.array: The softmax probabilities for each column, ensuring that the sum across rows for each column is 1.
    """
    x = x.astype(float)
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def softmax_derivative(softmax_output: np.array):
    """
    Computes the derivative of the softmax function.

    Args:
        softmax_output (np.array): The output of the softmax function.

    Returns:
        np.array: The Jacobian matrix of the softmax function
    """
    s = softmax_output.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

functions = {
    "ReLU": ReLU,
    "softmax": softmax
}

derivates = {
    "ReLU": ReLU_derivative,
    "softmax": softmax_derivative
}