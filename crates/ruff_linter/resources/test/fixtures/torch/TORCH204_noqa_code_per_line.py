import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10)]  # noqa: TORCH204
        self.others = {"fc": nn.Linear(10, 10)}
