"""Test cases for TORCH006: Use torch.inference_mode instead of torch.no_grad."""

import torch
import torch as th
from torch import no_grad


# Errors (should trigger TORCH006)


# Context manager
with torch.no_grad():
    pass


# Aliased import
with th.no_grad():
    pass


# `from torch import no_grad`
with no_grad():
    pass


# Decorator factory (call expression)
@torch.no_grad()
def predict(x):
    return x


# Bare call
ctx = torch.no_grad()


# should NOT trigger TORCH006


# Already using inference_mode
with torch.inference_mode():
    pass


# Different attribute on torch
with torch.cuda.device(0):
    pass


# Not a call on torch.no_grad — just a name with the same suffix
my_no_grad = 1
