import torch.nn as nn

nn.utils.clip_grad_value_([], clip_value=1.0)  # noqa: TORCH602
nn.utils.clip_grad_value_([], clip_value=2.0)
