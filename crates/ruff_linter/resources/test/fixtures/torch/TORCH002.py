"""Test cases for TORCH002: Avoid accessing tensor.data."""

import torch
import torch as th

# Errors (should trigger TORCH002)

# Basic tensor.data access
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.data

# On a variable assigned from aliased import
z = th.tensor([1.0])
w = z.data

# Chained with further attribute access
v = x.data.numpy()

# In an expression
result = x.data + 1

# Nested
foo(x.data)

# In a list comprehension
vals = [t.data for t in tensors]

# On a function return
output = model(input).data

# On subscript
batch = batches[0].data

# should NOT trigger TORCH002

# Assignment to .data (store context, not load)
x.data = torch.tensor([0.0])

# self.data and cls.data (common in non-tensor classes)
class MyDataset:
    def __init__(self):
        self.data = []

    def get(self):
        return self.data

    @classmethod
    def from_cls(cls):
        return cls.data

# Module-level access (torch.utils.data)
from torch.utils.data import DataLoader

# Calling .data() as a method (not attribute access)
x.data()

# Literals
{}.data
[].data
"string".data

# Type annotations are fine (not runtime access)
def process(x: torch.Tensor) -> torch.Tensor:
    return x.detach()
