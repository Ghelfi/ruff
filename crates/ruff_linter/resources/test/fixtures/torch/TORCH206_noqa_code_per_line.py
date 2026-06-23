import torch.nn as nn

nn.utils.weight_norm(nn.Linear(10, 10))  # noqa: TORCH206
nn.utils.weight_norm(nn.Linear(20, 20))
