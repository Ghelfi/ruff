"""Test cases for TORCH600: deprecated `torch.cuda.amp.autocast`."""

import torch
from torch.cuda.amp import autocast

# Errors (should trigger TORCH600)
with torch.cuda.amp.autocast():
    pass

with torch.cuda.amp.autocast(dtype=torch.float16):
    pass

with autocast():
    pass

# Should NOT trigger TORCH600
with torch.amp.autocast("cuda"):
    pass

with torch.amp.autocast("cpu"):
    pass
