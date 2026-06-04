"""Test cases for TORCH101: try/except inside a torch.compile-decorated function."""

import torch


# Errors (should trigger TORCH101)


@torch.compile
def bad_simple(x):
    try:
        return 1.0 / x
    except ZeroDivisionError:
        return torch.zeros_like(x)


@torch.compile(mode="default")
def bad_factory(x):
    try:
        return x.sum()
    except RuntimeError:
        return x


@torch.compile
def bad_try_finally(x):
    try:
        return x + 1
    finally:
        pass


@torch.compile
def bad_nested(x):
    if x.sum() > 0:
        try:
            return x
        except Exception:
            return x
    return x


# should NOT trigger TORCH101


def not_compiled(x):
    try:
        return x
    except Exception:
        return x


@torch.compile
def ok_no_try(x):
    return x + 1
