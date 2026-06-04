"""Test cases for TORCH012: Use .to() instead of .cpu()/.cuda()/.float()/etc."""

import torch


# Errors (should trigger TORCH012)


def to_device():
    x = torch.tensor([1.0])
    y1 = x.cpu()
    y2 = x.cuda()


def to_dtype():
    x = torch.tensor([1.0])
    y1 = x.float()
    y2 = x.double()
    y3 = x.half()
    y4 = x.long()
    y5 = x.int()
    y6 = x.short()
    y7 = x.byte()
    y8 = x.char()
    y9 = x.bool()


def with_args():
    # Has arguments — report but do not autofix.
    x = torch.tensor([1.0])
    y = x.cuda(device=0)


# should NOT trigger TORCH012


def ok_to():
    x = torch.tensor([1.0])
    y = x.to("cuda")
    z = x.to(torch.float32)


def ok_other_method():
    x = torch.tensor([1.0])
    y = x.detach()
    z = x.clone()


def ok_dunder():
    x = torch.tensor([1.0])
    y = x.__int__()  # dunder, ignored
