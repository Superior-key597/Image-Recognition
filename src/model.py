import numpy as np
from .activations import ReLU, ReLU_derivative, softmax
from .initializers import he_uniform

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.W = []
        self.b = []

        for i in range(len(layer_sizes) - 1):
            self.W.append(he_uniform(layer_sizes[i], layer_sizes[i + 1]))
            self.b.append(np.zeros((layer_sizes[i + 1], 1)))

    def forward(self, X):
        self.Z = []
        self.A = [X]

        for i in range(len(self.W) - 1):
            Z = self.W[i] @ self.A[-1] + self.b[i]
            A = ReLU(Z)

            self.Z.append(Z)
            self.A.append(A)

        Z = self.W[-1] @ self.A[-1] + self.b[-1]
        A = softmax(Z)

        self.Z.append(Z)
        self.A.append(A)

        return A
    
    def backward(self, Y):
        m = Y.shape[-1]

        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        dZ = self.A[-1] - Y

        for i in reversed(range(len(self.W))):
            dW[i] = (1 / m) * dZ @ self.A[i].T
            db[i] = np.sum(dZ, axis=1, keepdims=True)

            if i > 0:
                dZ = (self.W[i].T @ dZ) * ReLU_derivative(self.Z[i - 1])

        return dW, db
    
    def update(self, dW, db, lr):
        for i in range(len(self.W)):
            self.W[i] -= dW[i] * lr
            self.b[i] -= db[i] * lr
