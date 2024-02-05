import tinny
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_train = np.array(pd.read_csv('./data/train.csv')).T
data_test = np.array(pd.read_csv('./data/test.csv')).T

m, n = data_train.shape

X_train = data_train[1:n]
y_train = data_train[0]

X_train, X_test, y_train, y_test = train_test_split(X_train.T, y_train, train_size=35000, random_state=42)

X_train = X_train.T / 255.0
X_test = X_test.T / 255.0


l1 = tinny.DenseLayer(784, 10, "ReLU")
l2 = tinny.OutputLayer(10, 10, "softmax")
nn = tinny.TiNNyNetwork([l1, l2])

nn.train(X_train, y_train, iterations=350, learning_rate=0.01)
nn.test(X_test, y_test)