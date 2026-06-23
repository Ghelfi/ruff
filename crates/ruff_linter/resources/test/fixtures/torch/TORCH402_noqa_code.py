# ruff: noqa: TORCH402

import torch
from torch.nn.parallel import DistributedDataParallel

model = ...
DistributedDataParallel(torch.compile(model))
