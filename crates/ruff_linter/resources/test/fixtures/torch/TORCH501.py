"""Test cases for TORCH501: `zero_grad(set_to_none=False)`."""

import torch

optimizer = torch.optim.SGD([], lr=0.1)

# Errors (should trigger TORCH501)
optimizer.zero_grad(set_to_none=False)
model.zero_grad(set_to_none=False)

# Should NOT trigger TORCH501
optimizer.zero_grad()
optimizer.zero_grad(set_to_none=True)
