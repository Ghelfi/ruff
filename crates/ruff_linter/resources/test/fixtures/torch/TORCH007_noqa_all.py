# ruff: noqa

import torch


def f():
    x = torch.tensor([1.0], requires_grad=True)
    x.add_(1.0)
