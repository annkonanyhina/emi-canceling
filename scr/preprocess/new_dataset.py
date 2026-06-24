import torch
from torch.utils.data import Dataset as TorchDataset
import os

# собирает все .pt файлы в датасет

class Dataset(TorchDataset):
    def __init__(self, root, device):
        self.root = root
        self.device = device

        self.files = []
        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                if f.endswith(".pt"):
                    self.files.append(os.path.join(dirpath, f))

        self.files.sort()

    def __getitem__(self, index):
        data = torch.load(self.files[index], map_location=torch.device(self.device))

        features = data[:, 0, :].unsqueeze(1)
        target = data[:, 2, :].unsqueeze(1)

        return features, target

    def __len__(self):
        return len(self.files)
