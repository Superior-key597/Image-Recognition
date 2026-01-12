import numpy as np
from src.model import NeuralNetwork
from src.losses import cross_entropy_loss
from src.utils import one_hot, create_batches, accuracy

def train_nn(X, Y, hidden_layers, lr=0.01, batch_size=64, epochs=50, verbose=True, X_dev=None, Y_dev=None):
    num_features = X.shape[0]
    num_classes = len(np.unique(Y))

    model = NeuralNetwork([num_features] + hidden_layers + [num_classes])

    loss_history = []
    acc_history = []
    val_acc = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for X_batch, Y_batch in create_batches(X, Y, batch_size):
            Y_oh = one_hot(Y_batch, num_classes)

            A = model.forward(X_batch)

            loss = cross_entropy_loss(A, Y_oh)
            epoch_loss += loss

            preds = np.argmax(A, axis=0)
            acc = accuracy(preds, Y_batch)
            epoch_acc += acc

            dW, db = model.backward(Y_oh)
            model.update(dW, db, lr)

            num_batches += 1

        epoch_loss /= num_batches
        epoch_acc /= num_batches

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        if X_dev is not None:
            A_dev = model.forward(X_dev)
            val_preds = np.argmax(A_dev, axis=0)
            val_acc.append(accuracy(val_preds, Y_dev))

        print(f"Epoch {epoch+1:3d} | "
              f"Loss: {loss_history[-1]:.4f} | "
              f"Train Acc: {acc_history[-1]:.3f} | "
              f"Val Acc: {val_acc[-1] if val_acc else 0:.3f}")
            

    return model, loss_history, acc_history, val_acc




