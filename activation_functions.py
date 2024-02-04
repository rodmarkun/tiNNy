import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_derivative(x):
    return (x > 0).astype(int)

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def softmax_derivative(softmax_output):
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