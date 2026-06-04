# ruff: noqa

import torch


@torch.compile
def f(x):
    print(x)
    return x
