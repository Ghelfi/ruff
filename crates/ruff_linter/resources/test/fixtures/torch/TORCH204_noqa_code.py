# ruff: noqa: TORCH204

import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10)]
