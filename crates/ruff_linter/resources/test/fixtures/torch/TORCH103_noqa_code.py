# ruff: noqa: TORCH103

import torch


@torch.compile
def f(x):
    return x.item()
