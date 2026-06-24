import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from code.preprocess.new_dataset import Dataset

SLICE_WINDOW = 10
BATCH_SIZE = 64


def slicing(x):
    return x[..., SLICE_WINDOW:x.shape[-1] - SLICE_WINDOW]


def complex_rms_v(x):

    I = x[:, 0, :, :]
    Q = x[:, 1, :, :]

    rms = torch.sqrt(torch.mean(I**2 + Q**2, dim=(-2, -1)))
    return rms


def compute_all_std(root_path, device):
    dataset = Dataset(root_path, device=device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_std = []

    with torch.no_grad():
        for _, target in loader:
            target = slicing(target)
            std_batch = complex_rms_v(target)
            all_std.extend(std_batch.detach().cpu().numpy())

    return np.array(all_std), len(dataset)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_root = "./pre_processed_data/train"
    test_root  = "./pre_processed_data/test"

    train_std, train_size = compute_all_std(train_root, device)
    test_std, test_size   = compute_all_std(test_root, device)

    train_std_mv = train_std * 1e3
    test_std_mv = test_std * 1e3

    print("TRAIN size:", train_size)
    print("TEST  size:", test_size)

    print("TRAIN mean STD, mV:", np.mean(train_std_mv))
    print("TEST  mean STD, mV:", np.mean(test_std_mv))

    plt.figure(figsize=(8, 5))
    plt.hist(train_std_mv, bins=100, alpha=0.7)
    plt.title("Train")
    plt.xlabel("СКО комплексной огибающей, мВ")
    plt.ylabel("Количество примеров")
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(test_std_mv, bins=100, alpha=0.7)
    plt.title("Test")
    plt.xlabel("СКО комплексной огибающей, мВ")
    plt.ylabel("Количество примеров")
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(train_std_mv, bins=100, alpha=0.5, label="Train")
    plt.hist(test_std_mv, bins=100, alpha=0.5, label="Test")
    plt.title("Train vs Test")
    plt.xlabel("СКО комплексной огибающей, мВ")
    plt.ylabel("Количество примеров")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

