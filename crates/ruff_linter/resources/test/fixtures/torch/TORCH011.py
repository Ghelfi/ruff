"""Test cases for TORCH011: .clone() on a grad-tracked tensor without .detach()."""

import torch


# Errors (should trigger TORCH011)


def bad_basic():
    x = torch.tensor([1.0], requires_grad=True)
    snap = x.clone()


def bad_ann_assign():
    x: torch.Tensor = torch.tensor([1.0], requires_grad=True)
    snap = x.clone()


def bad_zeros():
    x = torch.zeros(4, requires_grad=True)
    snap = x.clone()


# should NOT trigger TORCH011


def ok_explicit_detach():
    x = torch.tensor([1.0], requires_grad=True)
    # Detach is on the receiver chain.
    snap = x.detach().clone()


def ok_no_grad():
    x = torch.tensor([1.0])
    snap = x.clone()


def ok_explicit_false():
    x = torch.tensor([1.0], requires_grad=False)
    snap = x.clone()


def ok_other_method():
    x = torch.tensor([1.0], requires_grad=True)
    # Method other than `.clone()`.
    snap = x.cpu()
