import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
torch.qr(x)  # noqa: TORCH208
torch.cholesky(x)
