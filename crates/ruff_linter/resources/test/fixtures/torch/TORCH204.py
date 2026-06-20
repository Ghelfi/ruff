"""Test cases for TORCH204: `nn.Module` stored in a plain `list`/`dict`."""

import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()

        # Errors (should trigger TORCH204)
        self.list_layers = [nn.Linear(10, 10), nn.ReLU()]
        self.comprehension = [nn.Linear(10, 10) for _ in range(3)]
        self.dict_layers = {"fc": nn.Linear(10, 10)}
        self.qualified = [torch.nn.Conv2d(3, 16, 3)]

        # should NOT trigger TORCH204
        self.registered = nn.ModuleList([nn.Linear(10, 10)])
        self.registered_dict = nn.ModuleDict({"fc": nn.Linear(10, 10)})
        self.single = nn.Linear(10, 10)
        self.params = [nn.Parameter(torch.zeros(3))]
        self.numbers = [1, 2, 3]
        self.refs = [self.single]
        self.functional = [nn.functional.relu]


# Not a `self` attribute.
layers = [nn.Linear(10, 10)]
