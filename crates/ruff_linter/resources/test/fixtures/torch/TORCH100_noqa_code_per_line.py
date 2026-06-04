import torch


@torch.compile
def f(x):
    print(x)  # noqa: TORCH100
    print(x)
    return x
