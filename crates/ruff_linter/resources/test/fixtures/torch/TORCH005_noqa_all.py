# ruff: noqa

import torch


def predict(model, x):
    with torch.no_grad():
        return model(x)
