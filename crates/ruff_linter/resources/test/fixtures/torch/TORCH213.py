"""Test cases for TORCH213: direct `.forward()` call."""

import torch
import torch.nn as nn


model = nn.Linear(10, 10)


def run(x):
    # Errors (should trigger TORCH213)
    a = model.forward(x)
    b = model.forward(x, y=1)
    c = self.submodule.forward(x)
    return a, b, c


class Wrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        # Error: calling a submodule's `forward` directly.
        return self.inner.forward(x)


class Derived(nn.Module):
    def forward(self, x):
        # OK: delegating to the base class implementation.
        return super().forward(x)


def ok(x):
    # OK: invoking the module as a callable.
    return model(x)
