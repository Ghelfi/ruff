"""Test cases for TORCH008: Explicit device missing in torch.tensor() call."""

import torch
import torch as th
from torch import tensor


# Errors (should trigger TORCH008)


x = torch.tensor([1.0, 2.0])
y = torch.tensor([1.0], dtype=torch.float32)
z = th.tensor([1.0])
w = tensor([1.0])

# Inside expressions
v = process(torch.tensor([1.0]))


# should NOT trigger TORCH008


# device= present
ok = torch.tensor([1.0], device="cuda")
ok2 = torch.tensor([1.0], device=torch.device("cpu"))
ok3 = torch.tensor([1.0], dtype=torch.float32, device="cuda")

# Different constructor (TORCH008 only targets torch.tensor)
zs = torch.zeros(4)
os = torch.ones(4)

# Different qualified name (uppercase Tensor is covered by TORCH001)
bad_caps = torch.Tensor([1.0])
