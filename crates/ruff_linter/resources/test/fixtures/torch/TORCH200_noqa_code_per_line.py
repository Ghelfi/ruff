import torch.nn as nn


class Suppressed(nn.Module):
    def __init__(self):  # noqa: TORCH200
        self.fc = nn.Linear(10, 10)


class Reported(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 10)
