import tinny
import pandas as pd
import numpy as np

data_train = np.array(pd.read_csv('./data/train.csv')).T
data_test = np.array(pd.read_csv('./data/test.csv')).T

m, n = data_train.shape

X_train = data_train[1:n]
y_train = data_train[0]

X_train = X_train / 255.0


l1 = tinny.DenseLayer(784, 10, "ReLU")
l2 = tinny.OutputLayer(10, 10, "softmax")
nn = tinny.TiNNyNetwork([l1, l2])

nn.train(X_train, y_train, iterations=300, learning_rate=0.01)
