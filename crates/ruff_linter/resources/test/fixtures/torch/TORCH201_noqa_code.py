# ruff: noqa: TORCH201

import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 10)
        super().__init__()
