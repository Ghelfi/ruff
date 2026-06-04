# ruff: noqa

import torch


@torch.compile
def f(x):
    try:
        return x
    except Exception:
        return x
