import torch

x = torch.tensor([1.0])
y = x.data  # noqa: TORCH002
y = x.data
