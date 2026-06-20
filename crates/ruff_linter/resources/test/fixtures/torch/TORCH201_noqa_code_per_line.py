import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 10)  # noqa: TORCH201
        self.bn = nn.BatchNorm1d(10)
        super().__init__()
