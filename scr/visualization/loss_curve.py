import numpy as np
import matplotlib.pyplot as plt

PATH = "./models_save/ver_1/loss_history.npz"

def main():
    data = np.load(PATH)
    train = data["train"]
    val = data["val"]

    epochs = np.arange(1, len(train) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, label="train loss")
    plt.plot(epochs, val, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
