# ruff: noqa: TORCH011

import torch


def f():
    x = torch.tensor([1.0], requires_grad=True)
    snap = x.clone()
