# ruff: noqa: TORCH101

import torch


@torch.compile
def f(x):
    try:
        return x
    except Exception:
        return x
