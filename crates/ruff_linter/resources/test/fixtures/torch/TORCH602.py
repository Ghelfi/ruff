"""Test cases for TORCH602: `clip_grad_value_` discouraged."""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_

params = []

# Errors (should trigger TORCH602)
nn.utils.clip_grad_value_(params, clip_value=1.0)
torch.nn.utils.clip_grad_value_(params, clip_value=1.0)
clip_grad_value_(params, clip_value=1.0)

# Should NOT trigger TORCH602
nn.utils.clip_grad_norm_(params, max_norm=1.0)
