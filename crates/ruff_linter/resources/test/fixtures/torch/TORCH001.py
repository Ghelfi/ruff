"""Test cases for TORCH001: Use torch.tensor() instead of torch.Tensor()."""

import torch
import torch as th
from torch import Tensor
from torch import Tensor as T

# Errors (should trigger TORCH001)

# Basic torch.Tensor() calls
x = torch.Tensor()
x = torch.Tensor([1, 2, 3])
x = torch.Tensor([[1, 2], [3, 4]])
x = torch.Tensor([1.0, 2.0, 3.0])

# With alias
x = th.Tensor()
x = th.Tensor([1, 2, 3])

# From import
x = Tensor()
x = Tensor([1, 2, 3])

# Aliased from-import
x = T()
x = T([1, 2, 3])

# With keyword arguments
x = torch.Tensor([1, 2, 3], devie="cpu")
x = th.Tensor([1, 2, 3], devie="cpu")

# Nested in expressions
y = foo(torch.Tensor([1, 2, 3]))
z = [torch.Tensor([i]) for i in range(10)]

# Multi-line call
x = torch.Tensor(
    [1, 2, 3],
)

# should NOT trigger TORCH001)

# Correct usage: torch.tensor()
x = torch.tensor([1, 2, 3])
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x = torch.tensor([1, 2, 3], device="cuda")
x = th.tensor([1, 2, 3])

# Other torch constructors are fine
x = torch.zeros(3, 4)
x = torch.ones(3, 4)
x = torch.empty(3, 4)
x = torch.randn(3, 4)
x = torch.arange(10)
x = torch.linspace(0, 1, 10)

# Type annotations using torch.Tensor and Tensor are fine (not a call)
def foo(x: torch.Tensor) -> torch.Tensor:
    return x

def bar(x: Tensor) -> Tensor:
    return x

# Attribute access without call is fine
TensorClass = torch.Tensor