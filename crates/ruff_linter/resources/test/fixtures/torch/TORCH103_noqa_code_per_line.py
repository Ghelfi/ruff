import torch


@torch.compile
def f(x):
    a = x.item()  # noqa: TORCH103
    b = x.item()
    return a + b
