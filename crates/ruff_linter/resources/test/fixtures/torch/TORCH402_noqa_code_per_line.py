import torch
from torch.nn.parallel import DistributedDataParallel

model = ...
DistributedDataParallel(torch.compile(model))  # noqa: TORCH402
DistributedDataParallel(torch.compile(model))
