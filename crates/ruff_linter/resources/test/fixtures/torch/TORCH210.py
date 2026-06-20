"""Test cases for TORCH210: `nn.Parameter` stored in a plain `list`/`dict`."""

import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()

        # Errors (should trigger TORCH210)
        self.list_params = [nn.Parameter(torch.randn(10)), nn.Parameter(torch.randn(5))]
        self.comprehension = [nn.Parameter(torch.randn(10)) for _ in range(3)]
        self.dict_params = {"w": nn.Parameter(torch.randn(10))}

        # should NOT trigger TORCH210
        self.registered = nn.ParameterList([nn.Parameter(torch.randn(10))])
        self.registered_dict = nn.ParameterDict({"w": nn.Parameter(torch.randn(10))})
        self.single = nn.Parameter(torch.randn(10))
        self.modules = [nn.Linear(10, 10)]
        self.numbers = [1, 2, 3]


# Not a `self` attribute.
params = [nn.Parameter(torch.randn(10))]
