# ruff: noqa

from torch.utils.data import DataLoader, DistributedSampler

dataset = []
DataLoader(dataset, sampler=DistributedSampler(dataset))
