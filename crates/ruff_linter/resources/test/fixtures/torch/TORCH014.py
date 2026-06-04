"""Test cases for TORCH014: torch.tensor(...).to(device) -> torch.tensor(..., device=...)."""

import torch


# Errors (should trigger TORCH014)


# Single positional device string
x1 = torch.tensor([1.0]).to("cuda")

# Device keyword
x2 = torch.tensor([1.0]).to(device="cuda")

# Aliased import
import torch as th

x3 = th.tensor([1.0]).to("cpu")

# Existing kwargs in torch.tensor
x4 = torch.tensor([1.0], dtype=torch.float32).to("cuda")

# Multiple-arg .to() — reported without autofix
x5 = torch.tensor([1.0]).to("cuda", torch.float32)


# should NOT trigger TORCH014


# device already in torch.tensor
ok1 = torch.tensor([1.0], device="cuda").to("cuda")

# .to() not on torch.tensor
y = some_tensor.to("cuda")

# torch.tensor without trailing .to()
z = torch.tensor([1.0])

# .to() with no args is benign
w = torch.tensor([1.0]).to()
