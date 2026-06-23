"""Test cases for TORCH604: `torch.load` without `map_location`."""

import torch
from torch import load

# Errors (should trigger TORCH604)
torch.load("checkpoint.pt")
torch.load("checkpoint.pt", weights_only=True)
load("checkpoint.pt")

# Should NOT trigger TORCH604
torch.load("checkpoint.pt", map_location="cpu")
torch.load("checkpoint.pt", "cpu")  # Positional map_location.
torch.load("checkpoint.pt", map_location="cuda:0", weights_only=True)
