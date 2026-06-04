import torch


def first():
    x = torch.tensor([1.0], requires_grad=True)
    x.add_(1.0)  # noqa: TORCH007


def second():
    x = torch.tensor([1.0], requires_grad=True)
    x.add_(1.0)
