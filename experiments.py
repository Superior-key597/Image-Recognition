import numpy as np
import pandas as pd
from src.model import NeuralNetwork
from train import train_nn
from plot import plot_curves
import os

os.makedirs("results", exist_ok=True)

data = pd.read_csv('train.csv').to_numpy()
np.random.seed(42)
np.random.shuffle(data)

data_dev = data[:1000]
data_train = data[1000:]

X_dev = (data_dev[:, 1:] / 255.).T  
Y_dev = data_dev[:, 0].astype(int)  

X_train = (data_train[:, 1:] / 255.).T  
Y_train = data_train[:, 0].astype(int)

hidden_layers = [128, 64]
model, loss, acc, val_acc = train_nn(X_train, Y_train, hidden_layers, epochs=20, lr=0.01, batch_size=64, X_dev=X_dev, Y_dev=Y_dev)

plot_curves(loss, acc, val_acc)