"""Test cases for TORCH007: In-place op on a leaf tensor that requires grad."""

import torch


# Errors (should trigger TORCH007)


def bad_add():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    x.add_(1.0)


def bad_mul():
    x = torch.tensor([1.0], requires_grad=True)
    x.mul_(2.0)


def bad_ann_assign():
    x: torch.Tensor = torch.tensor([1.0], requires_grad=True)
    x.zero_()


def bad_zeros():
    # Any constructor with requires_grad=True trips it.
    x = torch.zeros(4, requires_grad=True)
    x.fill_(0.5)


# should NOT trigger TORCH007


def ok_out_of_place():
    x = torch.tensor([1.0], requires_grad=True)
    x = x + 1.0


def ok_no_grad():
    x = torch.tensor([1.0])
    x.add_(1.0)


def ok_explicit_false():
    x = torch.tensor([1.0], requires_grad=False)
    x.add_(1.0)


def ok_detached():
    x = torch.tensor([1.0], requires_grad=True)
    # The in-place op is on a detached view, not the leaf.
    x.detach().add_(1.0)


def ok_dunder():
    x = torch.tensor([1.0], requires_grad=True)
    # Dunder method, not an in-place op.
    x.__init__()


def ok_short_name():
    x = torch.tensor([1.0], requires_grad=True)
    # Method `_` alone is too short / unusual; not an in-place op.
    x._()
