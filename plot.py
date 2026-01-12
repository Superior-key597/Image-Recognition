import matplotlib.pyplot as plt

def plot_curves(loss, train_acc, val_acc=None):
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("results/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    if val_acc:
        plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig("results/accuracy_curve.png")
    plt.close()