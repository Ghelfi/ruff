"""Test cases for TORCH402: `torch.compile` before DDP wrap."""

import torch
from torch.nn.parallel import DistributedDataParallel

model = ...

# Errors (should trigger TORCH402)
DistributedDataParallel(torch.compile(model))
DistributedDataParallel(torch.compile(model), device_ids=[0])

# Should NOT trigger TORCH402
torch.compile(DistributedDataParallel(model))
DistributedDataParallel(model)
