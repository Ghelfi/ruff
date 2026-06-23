# ruff: noqa: TORCH602

import torch.nn as nn

nn.utils.clip_grad_value_([], clip_value=1.0)
