import numpy as np

def he_uniform(n_in, n_out):
    limit = np.sqrt(6 / n_in)
    return np.random.uniform(-limit, limit, size=(n_out, n_in))