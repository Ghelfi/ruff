"""Test cases for TORCH207: `torch.arange` with a float step."""

import torch

# Errors (should trigger TORCH207)
torch.arange(0, 1, 0.1)
torch.arange(0.0, 5.0, 0.5)
torch.arange(0, 10, step=0.25)
torch.arange(0, 1, -0.1)

# should NOT trigger TORCH207
torch.arange(0, 10, 2)
torch.arange(10)
torch.arange(0, 10)
torch.arange(0, 10, step=2)
torch.arange(0.0, 1.0)  # Float bounds but integer (default) step.
torch.linspace(0, 1, 11)
