"""Test cases for TORCH206: deprecated `torch.nn.utils.weight_norm`."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# Errors (should trigger TORCH206)
nn.utils.weight_norm(nn.Linear(10, 10))
torch.nn.utils.weight_norm(nn.Linear(10, 10))
weight_norm(nn.Linear(10, 10))

# Should NOT trigger TORCH206
nn.utils.parametrizations.weight_norm(nn.Linear(10, 10))
torch.nn.utils.parametrizations.weight_norm(nn.Linear(10, 10))
