"""Test cases for TORCH102: data-dependent if on a tensor value."""

import torch


# Errors (should trigger TORCH102)


@torch.compile
def bad_compare_attr(x):
    y = torch.zeros(4)
    if y.sum() > 0:
        return x
    return x + 1


@torch.compile
def bad_direct_name(x):
    y = torch.tensor([1.0])
    if y:
        return x
    return x + 1


@torch.compile
def bad_call_on_tensor(x):
    y = torch.zeros(4)
    if y.any():
        return x
    return x + 1


@torch.compile(mode="default")
def bad_with_factory(x):
    y = torch.ones(4)
    if y.mean() > 0.5:
        return x
    return x + 1


# should NOT trigger TORCH102


@torch.compile
def ok_python_value(x, flag):
    if flag:  # flag is a function arg with no torch initializer
        return x
    return x + 1


@torch.compile
def ok_constant(x):
    if True:
        return x
    return x + 1


def not_compiled(x):
    y = torch.zeros(4)
    if y.sum() > 0:
        return x
    return x


@torch.compile
def ok_non_torch_name(x):
    y = some_other_lib()
    if y:
        return x
    return x + 1
