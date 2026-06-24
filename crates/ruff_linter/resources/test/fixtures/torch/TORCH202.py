"""Test cases for TORCH202: layer assigned to `self` outside `__init__`."""

import torch.nn as nn


class Bad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Error: rebuilds the layer each forward call.
        self.fc = nn.Linear(10, 10)
        return self.fc(x)

    def helper(self, x):
        # Error: assignment in any non-__init__ method counts.
        self.conv = nn.Conv2d(3, 3, 3)
        return self.conv(x)

    def annotated(self, x):
        # Error: annotated assignment is also flagged.
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(3)
        return self.bn(x)


class Good(nn.Module):
    def __init__(self):
        super().__init__()
        # OK: layers defined in __init__.
        self.fc = nn.Linear(10, 10)
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        # OK: assigning a non-layer value to self.
        self.last_output = self.fc(x)
        return self.last_output


class NotAModule:
    def forward(self, x):
        # OK: not an nn.Module subclass.
        self.fc = nn.Linear(10, 10)
        return x
