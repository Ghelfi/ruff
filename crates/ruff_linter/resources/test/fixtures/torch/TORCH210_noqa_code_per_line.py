import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = [nn.Parameter(torch.randn(10))]  # noqa: TORCH210
        self.others = {"w": nn.Parameter(torch.randn(10))}
