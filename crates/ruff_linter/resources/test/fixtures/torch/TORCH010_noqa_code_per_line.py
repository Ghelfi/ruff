import torch


def f():
    x = torch.zeros(4)
    y = torch.tensor(x)  # noqa: TORCH010
    z = torch.tensor(x)
