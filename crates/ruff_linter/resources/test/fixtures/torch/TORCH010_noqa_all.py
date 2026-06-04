# ruff: noqa

import torch


def f():
    x = torch.zeros(4)
    y = torch.tensor(x)
