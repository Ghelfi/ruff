"""Test cases for TORCH208: deprecated torch.* linalg functions."""

import torch
from torch import qr

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Errors (should trigger TORCH208)
torch.qr(x)
torch.cholesky(x)
torch.matrix_rank(x)
torch.pinverse(x)
torch.inverse(x)
torch.det(x)
torch.slogdet(x)
torch.matrix_power(x, 2)
torch.matrix_exp(x)
torch.symeig(x)  # No autofix: argument order differs.
torch.solve(x, x)  # No autofix: argument order differs.
torch.lstsq(x, x)  # No autofix: argument order differs.
torch.eig(x)  # No autofix: return shape differs.
qr(x)

# Should NOT trigger TORCH208
torch.linalg.qr(x)
torch.linalg.solve(x, x)
torch.matmul(x, x)
