import torch.nn as nn
import torch.nn.functional as F
import torch

# НОВОЕ
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.features = nn.Sequential(
          nn.Conv2d(2, 128, (1, 11), stride=1, padding=(0, 5)), 
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 64, (1, 9), stride=1, padding=(0, 4)),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 32, (1, 5), stride=1, padding=(0, 2)),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32, 32, 1, stride=1, padding=0),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          # stride=1, чтобы длина осталась 148
          nn.Conv2d(32, 2, (1, 7), stride=1, padding=(0, 3)), 
        )

    def forward(self, x):
        # на входе (B, 2, 1, 148)
        output = self.features(x) 
        return output # и на выходе (B, 2, 1, 148)

