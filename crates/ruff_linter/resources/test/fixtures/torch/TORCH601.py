"""Test cases for TORCH601: deprecated `torch.cuda.amp.GradScaler`."""

import torch
from torch.cuda.amp import GradScaler

# Errors (should trigger TORCH601)
scaler = torch.cuda.amp.GradScaler()
scaler = torch.cuda.amp.GradScaler(init_scale=2.0**16)
scaler = GradScaler()

# Should NOT trigger TORCH601
scaler = torch.amp.GradScaler("cuda")
scaler = torch.amp.GradScaler("cpu")
