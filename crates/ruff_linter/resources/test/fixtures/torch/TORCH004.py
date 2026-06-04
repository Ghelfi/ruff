"""Test cases for TORCH004: Missing .detach() before .numpy() / .item()."""

import torch

# Errors (should trigger TORCH004)

x = torch.tensor([1.0, 2.0], requires_grad=True)

# Basic .numpy() without detach
arr = x.numpy()

# Basic .item() without detach
value = x.item()

# Basic .tolist() without detach
values = x.tolist()

# Chained: .cpu() without detach
arr2 = x.cpu().numpy()

# Chained: subscript receiver
arr3 = tensors[0].item()

# Inside a larger expression
process(x.numpy())

# should NOT trigger TORCH004

# detach in the chain
arr_ok = x.detach().numpy()
arr_ok2 = x.detach().item()
arr_ok3 = x.detach().tolist()

# detach earlier in the chain
arr_ok4 = x.detach().cpu().numpy()
arr_ok5 = x.cpu().detach().numpy()

# unrelated method
x.size()
x.shape

# numpy/item on non-attribute callee (won't match)
numpy()
item()
