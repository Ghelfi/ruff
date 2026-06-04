# ruff: noqa

import torch


@torch.compile
def f(x):
    return x.item()
