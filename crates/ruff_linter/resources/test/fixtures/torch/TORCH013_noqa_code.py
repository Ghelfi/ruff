# ruff: noqa: TORCH013

import torch

x = torch.zeros(1, 3, 1)
y = x.squeeze()
