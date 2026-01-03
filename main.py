import numpy as np
import pandas as pd

def he_uniform(n_in, n_out):
    limit = np.sqrt(6 / n_in)
    return np.random.uniform(-limit, limit, size=(n_out, n_in))

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_derivative(Z):
    return Z > 0

def softmax(zs):
    zs -= np.max(zs, axis=0, keepdims=True)
    exp_z = np.exp(zs)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def init_params():
    W1 = he_uniform(784, 10); b1 = np.zeros((10, 1))
    W2 = he_uniform(10, 8); b2 = np.zeros((8, 1))
    W3 = he_uniform(8, output_size); b3 = np.zeros((output_size, 1))

    return W1, b1, W2, b2, W3, b3

def forward_prop(W1, b1, W2, b2, W3, b3, x):
    Z1 = W1.dot(x) + b1; A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2; A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3; A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backprop(Z1, A1, Z2, A2, W2, Z3, A3, W3, Y, X):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y
    dW3 = (1/m) * dZ3.dot(A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * ReLU_derivative(Z2)
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr):
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, lr):
    W1, b1, W2, b2, W3, b3 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backprop(Z1, A1, Z2, A2, W2, Z3, A3, W3, Y, X)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, 0.1)

        if i % 50 == 0:
            print("Iteration:", i)
            print("Accuracy:", get_accuracy(get_predictions(A3), Y))

    return W1, b1, W2, b2, W3, b3


data = pd.read_csv('train.csv')
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.

output_size = 10

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 500, 0.1)
