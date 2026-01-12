import numpy as np

def cross_entropy_loss(predicted, target):
    eps = 1e-15
    predicted = np.clip(predicted, eps, 1 - eps)
    m = target.shape[1]
    return -np.sum(target * np.log(predicted)) / m