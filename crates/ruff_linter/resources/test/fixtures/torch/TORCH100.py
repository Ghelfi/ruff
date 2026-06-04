"""Test cases for TORCH100: print() inside a torch.compile-decorated function."""

import torch


# Errors (should trigger TORCH100)


@torch.compile
def bad_bare_decorator(x):
    print(x)
    return x + 1


@torch.compile(mode="default")
def bad_decorator_factory(x):
    print("hello", x)
    return x + 1


@torch.compile
def bad_nested_print(x):
    if x.sum() > 0:
        print(x)
    return x


@torch.compile
def bad_inner_function(x):
    def inner():
        print(x)

    inner()
    return x


# should NOT trigger TORCH100


def not_compiled(x):
    print(x)
    return x + 1


@torch.compile
def ok_no_print(x):
    return x + 1


@torch.compile
def ok_other_call(x):
    torch.relu(x)
    return x


# Decorator from a different library — not torch.compile.
@something.compile
def ok_other_compile(x):
    print(x)
    return x
