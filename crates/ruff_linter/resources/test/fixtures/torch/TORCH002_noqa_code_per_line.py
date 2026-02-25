import torch

x = torch.tensor([1.0])
y = x.data  # ruff: noqa: TORCH002
y = x.data
