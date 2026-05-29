"""Test cases for TORCH003: Missing force=True when calling .numpy()."""

import torch
import torch as th

# Errors (should trigger TORCH003)

# Basic .numpy() without force
x = torch.tensor([1.0, 2.0])
arr = x.numpy()

# On aliased import result
y = th.tensor([1.0])
arr2 = y.numpy()

# On chained expression
arr3 = model(input).numpy()

# On subscript result
arr4 = tensors[0].numpy()

# With other keyword arguments but no force
arr5 = x.numpy(dtype="float32")

# Nested in expression
result = process(x.numpy())

# In list comprehension
arrs = [t.numpy() for t in tensors]

# Multi-line call with a trailing comma (exercises argument-insertion edge case)
arr7 = x.numpy(
    dtype="float32",
)

# should NOT trigger TORCH003

# Already has force=True
arr_ok = x.numpy(force=True)

# force=True with other args
arr_ok2 = x.numpy(dtype="float32", force=True)

# Explicit force=False — deliberate user opt-out, not a missing argument
arr_ok3 = x.numpy(force=False)

# Dynamically supplied force value — not safe to second-guess
arr_ok4 = x.numpy(force=use_force)

# Not a .numpy() call
x.tolist()
x.item()
x.detach()

# .numpy on non-torch objects (no torch import check bypassed, but
# we only gate on `torch` being imported, so these would still match
# if the object happens to have a .numpy() method — acceptable FP rate)
