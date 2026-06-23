import torch

optimizer = torch.optim.SGD([], lr=0.1)
optimizer.zero_grad(set_to_none=False)  # noqa: TORCH501
optimizer.zero_grad(set_to_none=False)
