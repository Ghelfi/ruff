# ruff: noqa: TORCH210

import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = [nn.Parameter(torch.randn(10))]
