import torch


def predict_first(model, x):
    with torch.no_grad():  # noqa: TORCH005
        return model(x)


def predict_second(model, x):
    with torch.no_grad():
        return model(x)
