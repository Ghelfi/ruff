"""Test cases for TORCH603: `torch.load` without `weights_only=True`."""

import torch
from torch import load

# Errors (should trigger TORCH603)
torch.load("checkpoint.pt")
torch.load("checkpoint.pt", map_location="cpu")
load("checkpoint.pt")

# Should NOT trigger TORCH603
torch.load("checkpoint.pt", weights_only=True)
torch.load("checkpoint.pt", weights_only=False)  # Already explicit.
torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
