# ruff: noqa: TORCH005

import torch


def predict(model, x):
    with torch.no_grad():
        return model(x)
