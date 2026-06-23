import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.fc = nn.Linear(10, 10)  # noqa: TORCH202
        self.conv = nn.Conv2d(3, 3, 3)
        return x
