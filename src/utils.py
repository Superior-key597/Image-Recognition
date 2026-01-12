import numpy as np

def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def create_batches(X, Y, batch_size):
    m = X.shape[1]
    indices = np.arange(m)
    np.random.shuffle(indices)

    X_shuffled = X[:, indices]
    Y_shuffled = Y[indices]

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]

        yield X_batch, Y_batch

def accuracy(preds, Y):
    return np.mean(preds==Y)
