# ruff: noqa: TORCH100

import torch


@torch.compile
def f(x):
    print(x)
    return x
