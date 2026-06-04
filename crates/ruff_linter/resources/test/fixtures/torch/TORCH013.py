"""Test cases for TORCH013: .squeeze() without dim."""

import torch


# Errors (should trigger TORCH013)


x = torch.zeros(1, 3, 1)
y1 = x.squeeze()

# Inside expression
y2 = process(x.squeeze())

# Chained
y3 = x.squeeze().sum()


# should NOT trigger TORCH013


# With positional dim
ok1 = x.squeeze(0)
ok2 = x.squeeze(-1)

# With keyword dim
ok3 = x.squeeze(dim=0)
ok4 = x.squeeze(dim=-1)

# Other method on tensor
ok5 = x.unsqueeze(0)
