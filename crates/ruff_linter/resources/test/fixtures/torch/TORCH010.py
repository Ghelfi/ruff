"""Test cases for TORCH010: torch.tensor(t) copies data; use t.clone()."""

import torch


# Errors (should trigger TORCH010)


def bad_from_zeros():
    x = torch.zeros(4)
    y = torch.tensor(x)


def bad_from_tensor():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor(x)


def bad_from_arange():
    x = torch.arange(4)
    y = torch.tensor(x)


# should NOT trigger TORCH010


def ok_from_list():
    # The argument is a literal, not a tensor.
    y = torch.tensor([1.0, 2.0])


def ok_no_init():
    # No binding source — function argument.
    def f(x):
        return torch.tensor(x)


def ok_from_non_torch():
    x = some_other_lib()
    y = torch.tensor(x)


def bad_with_kwargs():
    x = torch.zeros(4)
    # Single positional arg with kwargs is still the copy pattern.
    y = torch.tensor(x, dtype=torch.float32)


def ok_two_positional():
    x = torch.zeros(4)
    # Two positional args — not a tensor-copy call signature.
    y = torch.tensor(x, x)
