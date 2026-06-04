import torch


@torch.compile
def f(x):
    try:  # noqa: TORCH101
        return x
    except Exception:
        return x


@torch.compile
def g(x):
    try:
        return x
    except Exception:
        return x
