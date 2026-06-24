import os
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from new_dataset import Dataset
from shield_free_model import Net1
import numpy as np


class CustomMSELoss(nn.Module):
    def __init__(self, slice_window=10):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.slice_window = slice_window

    def forward(self, pred, target):
        # Обрезка краев
        pred_slice = pred[..., self.slice_window: pred.shape[-1] - self.slice_window]
        target_slice = target[..., self.slice_window: target.shape[-1] - self.slice_window]

        return self.mse_loss(pred_slice, target_slice)


def main():
    root = "./pre_processed_data/train"
    model_dir = "./models_save/ver_1"
    os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(0)

    best_val = float("inf")
    best_epoch = 0
    patience = 10
    min_delta = 1e-5
    no_improve = 0

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset = Dataset(root, device=device)
    print("dataset size:", len(dataset))

    n_total = len(dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_batch = 32
    val_batch = 32

    train_loader = DataLoader(train_set, train_batch, shuffle=True, drop_last=False)
    validation_loader = DataLoader(val_set, val_batch, shuffle=True, drop_last=False)

    print("train batches:", len(train_loader))
    print("val batches  :", len(validation_loader))

    model = Net1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = CustomMSELoss(slice_window=10)

    num_epochs = 100
    validation_idx = max(1, int(len(train_loader) / 2))

    train_epoch_losses = []
    val_epoch_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')

        model.train()
        batch_losses = []

        for i, training_pair in enumerate(tqdm(train_loader)):
            features, target = training_pair

            scale_constant = 5e-3
            features = features / scale_constant
            target = target / scale_constant


            optimizer.zero_grad()

            pred = model(features)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())


        avg_train = float(sum(batch_losses) / max(1, len(batch_losses)))
        train_epoch_losses.append(avg_train)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_features, val_target in validation_loader:
                scale_constant = 5e-3
                val_features = val_features / scale_constant
                val_target = val_target / scale_constant
                
                val_pred = model(val_features)
                val_loss = criterion(val_pred, val_target).item()
                val_losses.append(val_loss)

        avg_val = float(sum(val_losses) / max(1, len(val_losses)))
        val_epoch_losses.append(avg_val)

        print(f"avg train loss: {avg_train:.6f}")
        print(f"avg val   loss: {avg_val:.6f}")

        np.savez(
            os.path.join(model_dir, "loss_history.npz"),
            train=np.array(train_epoch_losses, dtype=np.float64),
            val=np.array(val_epoch_losses, dtype=np.float64),
        )

        if avg_val < best_val - min_delta:
            best_val = avg_val
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch={best_epoch}, best_val={best_val:.6f}")
                break

    return 0

if __name__ == '__main__':
    main()
