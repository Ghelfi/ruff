# ruff: noqa: TORCH010

import torch


def f():
    x = torch.zeros(4)
    y = torch.tensor(x)
