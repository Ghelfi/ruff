# ruff: noqa: TORCH303

from torch.utils.data import DataLoader

dataset = []
DataLoader(dataset, num_workers=4)
