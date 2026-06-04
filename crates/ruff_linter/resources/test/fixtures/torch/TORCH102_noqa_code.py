# ruff: noqa: TORCH102

import torch


@torch.compile
def f(x):
    y = torch.zeros(4)
    if y.sum() > 0:
        return x
    return x
