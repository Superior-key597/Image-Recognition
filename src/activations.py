import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    Z -= np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
