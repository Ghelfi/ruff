"""Test cases for TORCH504: `torch.cat` / `torch.stack` in a loop."""

import torch

xs = [torch.zeros(3)] * 5
result = torch.empty(0)

# Errors (should trigger TORCH504)
for x in xs:
    result = torch.cat([result, x])

for x in xs:
    result = torch.stack([result, x])

i = 0
while i < 5:
    result = torch.cat([result, xs[i]])
    i += 1

# Should NOT trigger TORCH504: call outside any loop.
result = torch.cat(xs)
result = torch.stack(xs)


def f(xs):
    # A function definition crosses the loop boundary; the inner `torch.cat`
    # is not part of the outer loop.
    return torch.cat(xs)


for x in xs:
    f(xs)
