import torch.nn as nn

model = nn.Linear(10, 10)


def run(x):
    a = model.forward(x)  # noqa: TORCH213
    b = model.forward(x)
    return a, b
