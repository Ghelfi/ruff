# ruff: noqa

import torch.nn as nn

model = nn.Linear(10, 10)


def run(x):
    return model.forward(x)
