"""Test cases for TORCH300: `DataLoader` with `DistributedSampler`."""

from torch.utils.data import DataLoader, DistributedSampler

dataset = []

# Errors (should trigger TORCH300)
DataLoader(dataset, sampler=DistributedSampler(dataset))
DataLoader(dataset, sampler=DistributedSampler(dataset), batch_size=32)

# Should NOT trigger TORCH300
DataLoader(dataset, sampler=DistributedSampler(dataset), drop_last=True)
DataLoader(dataset, sampler=DistributedSampler(dataset), drop_last=False)  # Explicit.
DataLoader(dataset)
DataLoader(dataset, batch_size=32)
