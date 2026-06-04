"""Test cases for TORCH103: .item() inside a torch.compile-decorated function."""

import torch


# Errors (should trigger TORCH103)


@torch.compile
def bad_simple(x):
    value = x.sum().item()
    return value + 1


@torch.compile(mode="default")
def bad_factory(x):
    return x.item()


@torch.compile
def bad_nested(x):
    if x.sum() > 0:
        return x.max().item()
    return 0


# should NOT trigger TORCH103


def not_compiled(x):
    return x.item()


@torch.compile
def ok_no_item(x):
    return x.sum()


@torch.compile
def ok_other_method(x):
    return x.tolist()


# .item() outside the compiled function — caller can extract.
def caller(x):
    return compiled(x).item()
