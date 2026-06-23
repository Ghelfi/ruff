# ruff: noqa

import torch

xs = [torch.zeros(3)] * 5
result = torch.empty(0)
for x in xs:
    result = torch.cat([result, x])
