"""Test cases for TORCH009: Device mismatch in a binary op."""

import torch


# Errors (should trigger TORCH009)


# Direct .cuda() + .cpu()
bad1 = a.cuda() + b.cpu()

# .to("cuda") + .to("cpu")
bad2 = a.to("cuda") + b.to("cpu")

# Device string with index
bad3 = a.to("cuda:0") + b.to("cpu")

# Reversed
bad4 = a.cpu() * b.cuda()

# Subtraction
bad5 = a.cuda() - b.cpu()


# should NOT trigger TORCH009


# Same device on both sides
ok1 = a.cuda() + b.cuda()
ok2 = a.cpu() + b.cpu()
ok3 = a.to("cuda") + b.to("cuda:1")

# Only one side specifies the device — the other is opaque.
ok4 = a.cuda() + b
ok5 = a + b.cpu()

# Neither side specifies the device.
ok6 = a + b

# .to() with non-string device.
ok7 = a.to("cuda") + b.to(device)
