# ruff: noqa

import torch.nn as nn

nn.utils.weight_norm(nn.Linear(10, 10))
