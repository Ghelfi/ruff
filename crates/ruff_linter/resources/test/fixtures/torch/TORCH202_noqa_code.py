# ruff: noqa: TORCH202

import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.fc = nn.Linear(10, 10)
        return self.fc(x)
